with open('cyclone_anomaly_analysis.py', 'r') as f:
    text = f.read()
import re
text = re.sub(r'if __name__ == "__main__\\:', '', text)  # clean up any mess
lines = text.split('\n')
out = []
in_main = False
for line in lines:
    if 'QUICK RUNNER (uncomment to execute)' in line:
        out.append(line)
        out.append('if __name__ == \"__main__\":')
        in_main = True
        continue
    if in_main:
        if line.strip() != '':
            out.append('    ' + line)
        else:
            out.append(line)
    else:
        out.append(line)

with open('cyclone_anomaly_analysis.py', 'w') as f:
    f.write('\n'.join(out))
