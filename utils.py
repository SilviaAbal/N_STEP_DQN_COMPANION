

def readTestTxt(filePath):

    lines = []

    # Open the file and read its lines
    with open(filePath, 'r') as file:
        for line in file:
            # Remove newline characters and append the line to the list
            clean_line = line.strip()
            lines.append(clean_line)
    
    return lines