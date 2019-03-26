import numpy as np


def generate_submission_file(file_name, classifier0, classifier1, classifier2,\
    Xte0, Xte1, Xte2):
    Yte0 = np.array(np.sign(classifier0.predict(Xte0)), dtype=int)
    Yte1 = np.array(np.sign(classifier1.predict(Xte1)), dtype=int)
    Yte2 = np.array(np.sign(classifier2.predict(Xte2)), dtype=int)

    # Map {-1, 1} back to {0, 1}
    Yte0[Yte0 == -1] = 0
    Yte1[Yte1 == -1] = 0
    Yte2[Yte2 == -1] = 0

    f = open(file_name, 'w')
    f.write("Id,Bound\n")
    count = 0
    for i in range(len(Yte0)):
        f.write(str(count)+","+str(Yte0[i])+"\n")
        count += 1
    for i in range(len(Yte1)):
        f.write(str(count)+","+str(Yte1[i])+"\n")
        count += 1
    for i in range(len(Yte2)):
        f.write(str(count)+","+str(Yte2[i])+"\n")
        count += 1
    f.close()
