a=int(input("Enter the terms: "))
s1=0                                         
s2=1 
if a<=0:
    print("The requested series is: ",s1)
else:
    print(s1,s2,end=" ")
    for x in range(2,a):
        next=s1+s2                           
        print(next,end=" ")
        s1=s2
        s2=next
# Enter the terms: 01123

