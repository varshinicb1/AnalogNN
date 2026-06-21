content = open('spice/pyspice_runner.py', 'r').read()
old = "analysis['op'].f'out_{i}'.voltage"
new = "analysis['op'][f'out_{i}'].voltage"
content = content.replace(old, new)
open('spice/pyspice_runner.py', 'w').write(content)
print("Fixed")
