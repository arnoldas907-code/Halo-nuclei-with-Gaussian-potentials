import numpy as np

def read_corenn_file(filename):
    blocks = []
    ann1 = -1.0/18.953

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip comments
            if line.startswith("#"):
                continue

            values = line.split()
            lunitary = (float(values[0]) < 0.0)
            if (lunitary):
               rnc=float(values[1]) # rnn/rnc
               blocks.append([4.0, rnc,0.0,0.0,float(values[2])])
               blocks.append([18.0,rnc,0.0,0.0,float(values[3])])
               blocks.append([72.0,rnc,0.0,0.0,float(values[4])])
               blocks.append([8.0, rnc,0.0,0.0,float(values[5])])
               blocks.append([6.0, rnc,0.0,0.0,float(values[6])])
               blocks.append([18.0,rnc,0.0,ann1,float(values[7])])
               if len(values)>8 :
                 blocks.append([4.0, rnc,0.0,ann1,float(values[8])])
                 blocks.append([8.0, rnc,0.0,ann1,float(values[9])])
               continue

            # Header line: 3 values
            if len(values) == 3:
                mc = float(values[0])
                rnn = float(values[1])
                rnc = float(values[2])
                continue

            # Data line: >= 6 values
            if len(values) >= 6:
                ab1 = 1.0/float(values[3])
                b3 = float(values[1])
                blocks.append([mc,rnc,ab1,0.0,b3])
#                blocks.append([1.0/mc,rnn/rnc,rnc/ab,0.0,b3])
                #continue

            # Data line: > 6 values
            if len(values) == 7:
                b3 = float(values[6])
                blocks.append([mc,rnc,ab1,ann1,b3])
#                blocks.append([1.0/mc,rnn/rnc,rnc/ab,rnn/ann,b3])

    return blocks

#print (np.pi, np.atan(1.0)*4.0)

if __name__ == "__main__":
    # test code only

  data = read_corenn_file("data.txt")

  print("Number of sets:", len(data))
  print("Data:", data)
