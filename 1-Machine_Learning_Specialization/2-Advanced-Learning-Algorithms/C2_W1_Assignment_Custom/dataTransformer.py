import numpy as np
import json
import matplotlib.pyplot as plt

X = np.load("data/X.npy")
y = np.load("data/y.npy")

X = X[0:1000]
y = y[0:1000]

print(X.shape)
print(y.shape)

XList = X.tolist()
XJson = json.dumps(XList)
f = open("X.json", "w")
f.write(XJson)
f.close()

yList = y.tolist()
yJson = json.dumps(yList)
f = open("y.json", "w")
f.write(yJson)
f.close()

# Save training set as images
i = 0
for x in X:
    x2020 = X[i].reshape(20, 20).T
    yScalar = y[i][0]
    plt.imshow(x2020, cmap='gray', aspect='equal')
    #plt.figimage(x2020, cmap='gray')
    plt.axis('off')
    # plt.show()
    i += 1
    fileName = "x_" + str(i) + "_" + str(yScalar) + ".png"
    plt.savefig(fileName, bbox_inches='tight', pad_inches=0, transparent=True, dpi=5.5)