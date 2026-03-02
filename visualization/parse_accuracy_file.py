
import re

def parse_accuracy_file(filepath):
    """
    Parse an accuracy file and extract all metrics.
    
    Returns:
        dict with keys:
            'overall_format': float
            'overall_reasoning': float
            'format_criteria': list of 7 floats
            'reasoning_criteria': list of 4 floats
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    result = {
        'overall_format': 0.0,
        'overall_reasoning': 0.0,
        'format_criteria': [],
        'reasoning_criteria': []
    }
    
    # Parse overall format accuracy
    match = re.search(r'Average Format Accuracy:\s+([\d.]+)%', content)
    if match:
        result['overall_format'] = float(match.group(1))
    
    # Parse overall reasoning accuracy
    match = re.search(r'Average Reasoning Format Accuracy:\s+([\d.]+)%', content)
    if match:
        result['overall_reasoning'] = float(match.group(1))
    
    # Parse format criteria (7 criteria)
    for i in range(1, 8):
        pattern = f'Format Criteria {i}.*?Accuracy:\\s+([\\d.]+)%'
        match = re.search(pattern, content)
        if match:
            result['format_criteria'].append(float(match.group(1)))
        else:
            result['format_criteria'].append(0.0)
    
    # Parse reasoning criteria (4 criteria)
    for i in range(1, 5):
        pattern = f'Reasoning Format Criteria {i}.*?Accuracy:\\s+([\\d.]+)%'
        match = re.search(pattern, content)
        if match:
            result['reasoning_criteria'].append(float(match.group(1)))
        else:
            result['reasoning_criteria'].append(0.0)
    
    return result


if __name__ == '__main__':
    # Test the parser
    import sys
    if len(sys.argv) > 1:
        result = parse_accuracy_file(sys.argv[1])
        print("Overall Format Accuracy:", result['overall_format'])
        print("Overall Reasoning Accuracy:", result['overall_reasoning'])
        print("Format Criteria:", result['format_criteria'])
        print("Reasoning Criteria:", result['reasoning_criteria'])
