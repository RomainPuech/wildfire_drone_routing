import time

start_time = time.time()
c = 0 
for i in range(100000000):
    c += 1
print("c: ", c)
end_time = time.time()
print("Time taken: ", end_time - start_time)