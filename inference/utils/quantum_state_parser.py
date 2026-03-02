import re
from typing import List, Optional

import numpy as np

from constants import COMMON_SQRT_VALUES


def parse_quantum_state(text: str) -> Optional[List[complex]]:
    """
    Extract quantum state array from text wrapped between <quantum_state> tags.
    Handles formats like: [1/√2, -0.71] or [0, 1] or [-1.00, 0] or arrays of any length
    Also converts symbolic values using COMMON_SQRT_VALUES mapping.

    Args:
        text: Text containing quantum state

    Returns:
        List of complex numbers representing the quantum state, or None if parsing fails
    """
    pattern = r'<quantum_state>\s*:?\s*\[(.*?)\]'
    match = re.search(pattern, text)

    if not match:
        return None

    state_str = match.group(1)
    components = [s.strip() for s in state_str.split(',')]

    if len(components) < 1:
        return None

    try:
        parsed_components = []
        for comp in components:
            value = parse_component_value(comp)
            if value is None:
                return None
            parsed_components.append(complex(value, 0))

        return parsed_components

    except (ValueError, AttributeError):
        return None


def parse_real_value(value_str: str) -> Optional[float]:
    """
    Parse a real number that might be symbolic (e.g., "1/√2", "√3/2", "1/8") or numeric.

    Args:
        value_str: String representation of a real number

    Returns:
        Float value or None if parsing fails
    """
    value_str = value_str.strip()

    is_negative = False
    if value_str.startswith('-'):
        is_negative = True
        value_str = value_str[1:].strip()
    elif value_str.startswith('+'):
        value_str = value_str[1:].strip()

    for numeric_value, symbolic_str in COMMON_SQRT_VALUES.items():
        if value_str == symbolic_str:
            return -numeric_value if is_negative else numeric_value

    sqrt_fraction_match = re.fullmatch(
        r'([0-9]*\.?[0-9]+)\s*/\s*√\s*([0-9]*\.?[0-9]+)',
        value_str
    )
    if sqrt_fraction_match:
        try:
            numerator = float(sqrt_fraction_match.group(1))
            radicand = float(sqrt_fraction_match.group(2))
            if radicand > 0:
                result = numerator / np.sqrt(radicand)
                return -result if is_negative else result
        except ValueError:
            pass

    sqrt_den_match = re.fullmatch(
        r'([0-9]*\.?[0-9]+)\s*/\s*\(?\s*([0-9]*\.?[0-9]+)?\s*√\s*([0-9]*\.?[0-9]+)\s*\)?',
        value_str
    )
    if sqrt_den_match:
        try:
            numerator = float(sqrt_den_match.group(1))
            coeff_str = sqrt_den_match.group(2)
            coeff = float(coeff_str) if coeff_str else 1.0
            radicand = float(sqrt_den_match.group(3))

            denominator = coeff * np.sqrt(radicand)
            if denominator != 0:
                result = numerator / denominator
                return -result if is_negative else result
        except ValueError:
            pass

    sqrt_num_match = re.fullmatch(
        r'([0-9]*\.?[0-9]+)?\s*√\s*([0-9]*\.?[0-9]+)\s*/\s*([0-9]*\.?[0-9]+)',
        value_str
    )
    if sqrt_num_match:
        try:
            coeff_str = sqrt_num_match.group(1)
            coeff = float(coeff_str) if coeff_str else 1.0
            radicand = float(sqrt_num_match.group(2))
            denominator = float(sqrt_num_match.group(3))
            if radicand > 0 and denominator != 0:
                result = (coeff * np.sqrt(radicand)) / denominator
                return -result if is_negative else result
        except ValueError:
            pass

    sqrt_num_paren_match = re.fullmatch(
        r'\(\s*([0-9]*\.?[0-9]+)?\s*√\s*([0-9]*\.?[0-9]+)\s*\)\s*/\s*([0-9]*\.?[0-9]+)',
        value_str
    )
    if sqrt_num_paren_match:
        try:
            coeff_str = sqrt_num_paren_match.group(1)
            coeff = float(coeff_str) if coeff_str else 1.0
            radicand = float(sqrt_num_paren_match.group(2))
            denominator = float(sqrt_num_paren_match.group(3))
            if radicand > 0 and denominator != 0:
                result = (coeff * np.sqrt(radicand)) / denominator
                return -result if is_negative else result
        except ValueError:
            pass

    try:
        result = float(value_str)
        return -result if is_negative else result
    except ValueError:
        pass

    if '/' in value_str and '√' not in value_str:
        try:
            parts = value_str.split('/')
            if len(parts) == 2:
                numerator = float(parts[0].strip())
                denominator = float(parts[1].strip())
                if denominator != 0:
                    result = numerator / denominator
                    return -result if is_negative else result
        except ValueError:
            pass

    return None


def parse_component_value(comp: str) -> Optional[complex]:
    """
    Parse a single component value, handling symbolic representations and complex numbers.

    Args:
        comp: Component string (e.g., "1/√2", "-0.71", "√3/2", "1/2", "0.5+0.3j", "√3/2j", "1/8i", "-0.71 + 1/√2j")

    Returns:
        Complex number or None if parsing fails
    """
    comp = comp.strip()
    comp = comp.replace('i', 'j')

    for i in range(1, len(comp)):
        if comp[i] in ['+', '-']:
            real_part = comp[:i].strip()
            imag_part = comp[i:].strip()

            if imag_part.endswith('j'):
                try:
                    real_value = parse_real_value(real_part)
                    if real_value is None:
                        continue

                    imag_coeff = imag_part[:-1].strip()

                    if not imag_coeff or imag_coeff == '+':
                        imag_value = 1.0
                    elif imag_coeff == '-':
                        imag_value = -1.0
                    else:
                        imag_value = parse_real_value(imag_coeff)
                        if imag_value is None:
                            continue

                    return complex(real_value, imag_value)
                except (ValueError, AttributeError):
                    continue

    if comp.endswith('j'):
        coeff_str = comp[:-1].strip()

        if not coeff_str or coeff_str == '+':
            return complex(0, 1)
        if coeff_str == '-':
            return complex(0, -1)

        coeff_value = parse_real_value(coeff_str)
        if coeff_value is not None:
            return complex(0, coeff_value)
        print(f"Failed to parse imaginary coefficient: {coeff_str}")
        return None

    real_value = parse_real_value(comp)
    if real_value is not None:
        return complex(real_value, 0)

    print(f"Failed to parse component value: {comp}")
    return None


def parse_all_quantum_states(text: str) -> List[Optional[List[complex]]]:
    """
    Extract all quantum state arrays from text (for multiple steps).
    Handles multiple formats:
    1. <quantum_state>: [...]
    2. "The current state becomes [...]" or "The current state becomes: [...]"
    3. "Current state: [...]"

    Detects which pattern exists in the text and uses only that pattern for parsing.
    Only returns successfully parsed states (skips those that fail to parse).

    Args:
        text: Text containing multiple quantum states

    Returns:
        List of quantum states (only successfully parsed states, no None values)
    """
    states = []

    pattern1 = r'<quantum_state>\s*:?\s*\[(.*?)\]'

    selected_pattern = None
    if re.search(pattern1, text):
        selected_pattern = pattern1

    if selected_pattern is None:
        return states

    matches = re.findall(selected_pattern, text)

    for match_content in matches:
        components = [s.strip() for s in match_content.split(',')]
        if len(components) < 1:
            continue

        try:
            parsed_components = []
            parse_success = True
            for comp in components:
                value = parse_component_value(comp)
                if value is None:
                    print(f"Failed to parse component '{comp}'")
                    parse_success = False
                    break
                parsed_components.append(value)

            if parse_success:
                states.append(parsed_components)
            else:
                print(f"Skipping state due to parse failure: {match_content}")
        except (ValueError, AttributeError):
            print(f"Exception occurred while parsing state: {match_content}")
            continue

    return states