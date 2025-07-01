a <- 10
b <- 5

jumlah <- a + b
kurang <- a - b
kali   <- a * b
bagi   <- a / b

Mencoba <-- jumlah * (kurang * kali) + a^4 / (bagi * kurang) * a^2 + b^3 

capture.output( Mencoba, file = "C:/Users/Asusg/Desktop/hasil.txt")
print(jumlah)
print(kurang)
print(kali)
print(bagi)
print(Mencoba)
