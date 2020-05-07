limit =int(input("Enter the terms: "))
s1=0                                         
s2=1 
for x in range(limit):
    print(s1)
    s=s1
    s1=s2
    s2=s+s1

