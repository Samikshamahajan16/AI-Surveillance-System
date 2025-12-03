import winsound

for i in range(5):
    print("Beep", i)
    winsound.Beep(1000, 300)
    winsound.Beep(1500, 300)
