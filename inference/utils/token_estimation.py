OVERHEAD = 30
STATE_LEN = 10

def estimate_max_tokens():
    for i in range(1, 7):
        for j in [10,20,30,40,50]:
            print(f"Qubits: {i}, Depth: {j}, Estimated Max Tokens: {(OVERHEAD + STATE_LEN * 2**i) * j}")

if __name__ == "__main__":
    estimate_max_tokens()