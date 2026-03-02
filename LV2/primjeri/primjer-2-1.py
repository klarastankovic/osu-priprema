import numpy as np

a = np.array([6, 2, 9]) #napravi polje od tri elementa
print(type(a)) #prikazi tip polja
print(a.shape) #koliko redaka ima vektor
print(a[0], a[1], a[2]) #prikazi prvi, drugi i treci element
a[1] = 5 #promijeni vrijednost polja na drugom mjestu
print(a) #prikazi cijeli a
print(a[1:2]) #izdvajanje
print(a[1:-1]) #izdvajanje

b = np.array([[3,7,1],[4,5,6]]) #napravi 2 dimenzionalno polje (matricu)
print(b.shape) #ispisi dimenzije polja
print(b) #ispisi cijelo polje b
print(b[0, 2], b[0, 1], b[1, 1]) #ispisi neke elemente polja
print(b[0:2,0:1]) #izdvajanje
print(b[:,0]) #izdvajanje

c = np.zeros((4,2)) #polje sa svim elementima jednakim 0
d = np.ones((3,2)) #polje sa svim elementima jednakim 1
e = np.full((1,2),5) #polje sa svim elementima jednakim 5
f = np.eye(2) #jedinicna matrica 2x2
g = np.array([1, 2, 3], np.float32)
duljina = len(g)
print(duljina)
h = g.tolist()
print(h)
c = g.transpose()
print(g)
np.concatenate((a, g,))