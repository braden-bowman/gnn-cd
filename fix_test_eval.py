with open('unsw_test_evaluation.py', 'r') as file:
    lines = file.readlines()

for i in range(len(lines)):
    if r'\\!=' in lines[i]:
        print(f"Found problematic line {i+1}: {lines[i].strip()}")
        lines[i] = lines[i].replace(r'\\!=', '\!=')
        print(f"Fixed to: {lines[i].strip()}")

with open('unsw_test_evaluation.py', 'w') as file:
    file.writelines(lines)

print("File fixed\!")
