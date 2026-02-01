import time
start_time = time.time()
a = 0
for i in range(60*60):
    for j in range(60*60):
        a += 1
print(a)
end_time = time.time()
print("Time taken: ", end_time - start_time)