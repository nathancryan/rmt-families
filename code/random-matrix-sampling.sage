def QR_householderO(a):
    n=a.dimensions()[0]
    A1=Matrix(RDF,n,n,a)
    A2=Matrix(RDF,n,n,a)
    lst=[]
    for i in range(n-1):
        mm=Matrix(RDF,n-i,n-i,0)
        for a in range(n-i):
            for b in range(n-i):
                mm[a,b]=A1[a+i,b+i]
        vv=[mm[k,0] for k in range(n-i)]
        xx=Matrix(RDF,n-i,1,vv)
        ee1=Matrix(RDF,n-i,n-i,identity_matrix(n-i))
        ee=Matrix(RDF,n-i,1,[ee1[k,0]for k in range(n-i)])
        aa=sqrt(sum([xx[j][0]**2 for j in range(n-i)]))
        uu=xx-aa*ee
        vv=uu/uu.norm()
        Q=identity_matrix(n-i)-2*vv*vv.transpose()
        Q = block_matrix([[identity_matrix(i),0],[0,Q]])
        lst.append(Q)
        A1=Q*A1
    for i in range(len(lst)):
        lst[i]=lst[i].transpose()
    Q=Matrix(RDF,n,n,identity_matrix(n))
    for i in range(len(lst)):
        Q=Q*lst[i]
    R=Q.transpose()*A2
    return Q,R

def QR_householderU(a):
    n = a.dimensions()[0]
    A1 = Matrix(CDF,n,n,a)
    A2 = Matrix(CDF,n,n,a)
    R = Matrix(CDF,n,n,identity_matrix(n))
    for i in range(n-1):
        xx = A1[range(i,n),i]
        theta = xx[0] / xx[0].norm()
        ee1 = Matrix(CDF,n-i,n-i,identity_matrix(n-i))
        xx = xx / xx.norm()
        uu = xx - theta[0]*ee1[range(n-i),0]
        vv = uu / uu.norm()
        Q = -theta[0].conjugate() * (ee1-2*vv*vv.conjugate_transpose())
        Q = block_matrix(CDF,2,2,[[Matrix(CDF,i,i,identity_matrix(i)),Matrix(CDF,i,n-i)],[Matrix(CDF,n-i,i),Q]],sparse = true)
        R = Q*R
        A1 = Q*A1
    Q = R.conjugate_transpose()
    R = R*A2
    return Q,R

def qnorm(h):
    Q.<i,j,k> = QuaternionAlgebra(RDF,-1,-1)
    return(sqrt(RR(sum(h.elementwise_product(h.conjugate()))[0].coefficient_tuple()[0])))
def QR_householderUSp(a):
    Q.<i,j,k> = QuaternionAlgebra(RDF,-1,-1)
    n = a.dimensions()[0]
    A1 = Matrix(Q,n,n,a)
    A2 = Matrix(Q,n,n,a)
    R = Matrix(Q,n,n,identity_matrix(n))
    for l in range(n-1):
        xx = A1[range(l,n),l]
        theta = xx[0]/qnorm(Matrix(Q,1,xx[0]))
        ee1 = Matrix(Q,n-l,n-l,identity_matrix(n-l))
        xx = xx / qnorm(xx)
        uu = xx - theta[0]*ee1[range(n-l),0]
        vv = uu / qnorm(uu)
        Qf = -theta[0].conjugate() * (ee1-2*vv*vv.conjugate_transpose())
        Qf = block_matrix(Q,2,2,[[Matrix(Q,l,l,identity_matrix(l)),0],[0,Qf]])
        R = Qf*R
        A1 = Qf*A1
    Qf = R.conjugate_transpose()
    R = R*A2
    return Qf,R

def myhaar_measureSO(n):
    m = Matrix(RDF, n, lambda i,j: normalvariate(0, 1))
    lbd = Matrix(RDF,n,lambda i,j:0)
    q,r = QR_householderO(m)
    for i in range (n):
        lbd[i,i] = r.diagonal()[i]/abs(r.diagonal()[i])
    if (det(q*lbd).real().round() ==1):
       aux = (q*lbd).eigenvalues()
       for i in range (n):
          aux[i] = aux[i].arg()
       aux.sort()
       return(aux)
    else:
        I = identity_matrix(n)
        I[0,0] = -1
        aux = (q*lbd*I).eigenvalues()
        for i in range (n):
            aux[i] = aux[i].arg()
        aux.sort()
        return(aux)

def myhaar_measureU(n):
    m = Matrix(CDF, n, lambda i,j: normalvariate(0, 1) + 1j*normalvariate(0, 1))
    lbd = Matrix(CDF,n,lambda i,j:0)
    q,r = QR_householderU(m)
    for i in range (n):
        lbd[i,i] = r.diagonal()[i]/abs(r.diagonal()[i])
    aux = (q*lbd).eigenvalues()
    for i in range (n):
        aux[i] = aux[i].arg()
    aux.sort()
    return(aux)

def myhaar_measureUSp(n):
    Q.<i,j,k> = QuaternionAlgebra(RDF,-1,-1)
    m = Matrix(Q,n,lambda a,b: normalvariate(0,1)+i*normalvariate(0,1)+j*normalvariate(0,1)+k*normalvariate(0,1))
    lbd = Matrix(Q,n)
    q,r = QR_householderUSp(m)
    for i in range (n):
        lbd[i,i] = r.diagonal()[i]/qnorm(Matrix(Q,1,r.diagonal()[i]))
    q = (q*lbd)
    I2 = identity_matrix(2)
    e1 = Matrix(CDF,2,[1j,0,0,-1j])
    e2 = Matrix(CDF,2,[0,1,-1,0])
    e3 = Matrix(CDF,2,[0,1j,1j,0])
    Q0 = Matrix(RDF,n)
    Q1 = Matrix(RDF,n)
    Q2 = Matrix(RDF,n)
    Q3 = Matrix(RDF,n)
    for i in range(n):
        for j in range(n):
            Q0[i,j] = q[i,j].coefficient_tuple()[0]
            Q1[i,j] = q[i,j].coefficient_tuple()[1]
            Q2[i,j] = q[i,j].coefficient_tuple()[2]
            Q3[i,j] = q[i,j].coefficient_tuple()[3]
    Qh = Q0.tensor_product(I2) + Q1.tensor_product(e1) + Q2.tensor_product(e2) + Q3.tensor_product(e3)
    aux = Qh.eigenvalues()
    for i in range (2*n):
        aux[i] = aux[i].arg()
    aux.sort()
    return(aux)

def positivemin(l):
    i=0
    e=-1
    while((e <= 0) & (i < len(l))):
        e = l[i]
        i+=1
    if (e<=0):
        e = 2*pi.n() + min(l)
    return e

#exU4=list()
#exU4_leastEigenvalue=list()
#exU12=list()
#exU12_leastEigenvalue=list()
#exSO4=list()
#exSO4_leastEigenvalue=list()
#exSO12=list()
#exSO12_leastEigenvalue=list()
#exUSp2=list()
#exUSp2_leastEigenvalue=list()
#exUSp6=list()
#exUSp6_leastEigenvalue=list()


#for i in range(10000):
#    husp2 = myhaar_measureUSp(2)
#    husp6 = myhaar_measureUSp(6)
    
    
#    hSO4 = myhaar_measureSO(4)
#    hSO12 = myhaar_measureSO(12)

#    hU4= myhaar_measureU(4)
#    hU12=myhaar_measureU(12)

#    exU4.append(hU4)
#    exU4_leastEigenvalue.append(positivemin(hU4))
#    exU12.append(hU12)
#    exU12_leastEigenvalue.append(positivemin(hU12))
#    exSO4.append(hSO4)
#    exSO4_leastEigenvalue.append(positivemin(hSO4))
#    exSO12.append(hSO12)
#    exSO12_leastEigenvalue.append(positivemin(hSO12))
#    exUSp2.append(husp2)
#    exUSp2_leastEigenvalue.append(positivemin(husp2))
#    exUSp6.append(husp6)
#    exUSp6_leastEigenvalue.append(positivemin(husp6))

#with open('SO4-large.csv', 'w') as archivo:
#    for i in range(len(exSO4)):
#        for j in range(len(exSO4[0])):
#            archivo.write(str(exSO4[i][j]))
#	    archivo.write("\n")
#            #if ((i+1)*(j+1)<len(exSO4)*len(exSO4[0])):
#            #    archivo.write(',')
#with open('SO12-large.csv', 'w') as archivo:
#    for i in range(len(exSO12)):
#        for j in range(len(exSO12[0])):
#            archivo.writeline(str(exSO12[i][j]))
#	    archivo.write("\n")
#            #if ((i+1)*(j+1)<len(exSO12)*len(exSO12[0])):
#            #    archivo.write(',')
#with open('exU4.csv', 'w') as archivo:
#    for i in range(len(exU4)):
#        for j in range(len(exU4[0])):
#            archivo.write(str(exU4[i][j]))
#	    archivo.write("\n")
#            #if ((i+1)*(j+1)<len(exU4)*len(exU4[0])):
#            #    archivo.write(',')
#with open('exU12.csv', 'w') as archivo:
#    for i in range(len(exU12)):
#        for j in range(len(exU12[0])):
#            archivo.write(str(exU12[i][j]))
#	    archivo.write("\n")
#            #if ((i+1)*(j+1)<len(exU12)*len(exU12[0])):
#                #archivo.write(',')
#with open('exUSp2.csv', 'w') as archivo:
#    for i in range(len(exUSp2)):
#        for j in range(len(exUSp2[0])):
#            archivo.write(str(exUSp2[i][j]))
#	    archivo.write("\n")
#            #if ((i+1)*(j+1)<len(exUSp2)*len(exUSp2[0])):
#                #archivo.write(',')
#with open('exUSp6.csv', 'w') as archivo:
#    for i in range(len(exUSp6)):
#        for j in range(len(exUSp6[0])):
#            archivo.write(str(exUSp6[i][j]))
#	    archivo.write("\n")
#            #if ((i+1)*(j+1)<len(exUSp6)*len(exUSp6[0])):
#            #   archivo.write(',')

#with open('SO4_leastEigenvalue.csv', 'w') as archivo:
#    for i in range(len(exSO4_leastEigenvalue)):
#            archivo.write(str(exSO4_leastEigenvalue[i]))
#	    archivo.write("\n")
#            #if (i<len(exSO4_leastEigenvalue)-1):
#            #    archivo.write(',')
#with open('SO12_leastEigenvalue.csv', 'w') as archivo:
#    for i in range(len(exSO12_leastEigenvalue)):
#            archivo.write(str(exSO12_leastEigenvalue[i]))
#	    archivo.write("\n")
#            #if (i<len(exSO12_leastEigenvalue)-1):
#            #    archivo.write(',')
#with open('exU4_leastEigenvalue.csv', 'w') as archivo:
#    for i in range(len(exU4_leastEigenvalue)):
#            archivo.write(str(exU4_leastEigenvalue[i]))
#	    archivo.write("\n")
#            #if (i<len(exU4_leastEigenvalue)-1):
#            #    archivo.write(',')
#with open('exU12_leastEigenvalue.csv', 'w') as archivo:
#    for i in range(len(exU12_leastEigenvalue)):
#            archivo.write(str(exU12_leastEigenvalue[i]))
#	    archivo.write("\n")
#            #if (i<len(exU12_leastEigenvalue)-1):
#            #    archivo.write(',')
#with open('exUSp2_leastEigenvalue.csv', 'w') as archivo:
#    for i in range(len(exUSp2_leastEigenvalue)):
#            archivo.write(str(exUSp2_leastEigenvalue[i]))
#	    archivo.write("\n")
#            #if (i<len(exUSp2_leastEigenvalue)-1):
#            #    archivo.write(',')
#with open('exUSp6_leastEigenvalue.csv', 'w') as archivo:
#    for i in range(len(exUSp6_leastEigenvalue)):
#            archivo.write(str(exUSp6_leastEigenvalue[i]))
#	    archivo.write("\n")
#            #if (i<len(exUSp6_leastEigenvalue)-1):
#            #    archivo.write(',')


def generate_eigenangles(filename, N, sample_size, group_as_string):
   if (group_as_string == 'symplectic'):
      ex=list()
      for i in range (sample_size):
         h=myhaar_measureUSp(N)
         ex.append(h)
      with open(''+filename+'.csv', 'w') as archivo:
         for i in range(len(ex)):
            for j in range(len(ex[0])):
               archivo.write(str(ex[i][j]))
               archivo.write("\n")
   elif(group_as_string == 'unitary'):
      ex=list()
      for i in range (sample_size):
         h=myhaar_measureU(N)
         ex.append(h)
      with open(''+filename+'.csv', 'w') as archivo:
         for i in range(len(ex)):
            for j in range(len(ex[0])):
               archivo.write(str(ex[i][j]))
               archivo.write("\n")
   elif(group_as_string == 'orthogonal'):
      ex=list()
      for i in range (sample_size):
         h=myhaar_measureSO(N)
         ex.append(h)
      with open(''+filename+'.csv', 'w') as archivo:
         for i in range(len(ex)):
            for j in range(len(ex[0])):
               archivo.write(str(ex[i][j]))
               archivo.write("\n")
   else:
      return 'Not specified group'

#generate_eigenangles("Sp50", 50, 1000, "symplectic")


def generate_lowest_eigenangles(filename, N, sample_size, group_as_string):
   if (group_as_string == 'symplectic'):
      ex=list()
      ex_lowest_eigenvalue=list()
      for i in range (sample_size):
         h=myhaar_measureUSp(N)
         ex.append(h)
         ex_lowest_eigenvalue.append(positivemin(h))
      with open(''+filename+'.csv', 'w') as archivo:
         for i in range(len(ex_lowest_eigenvalue)):
            for j in range(len(ex_lowest_eigenvalue[0])):
               archivo.write(str(ex_lowest_eigenvalue[i][j]))
               archivo.write("\n")
   elif(group_as_string == 'unitary'):
      ex=list()
      ex_lowest_eigenvalue=list()
      for i in range (sample_size):
         h=myhaar_measureU(N)
         ex.append(h)
         ex_lowest_eigenvalue.append(positivemin(h))
      with open(''+filename+'.csv', 'w') as archivo:
         for i in range(len(ex_lowest_eigenvalue)):
            for j in range(len(ex_lowest_eigenvalue[0])):
               archivo.write(str(ex_lowest_eigenvalue[i][j]))
               archivo.write("\n")
   elif(group_as_string == 'orthogonal'):
      ex=list()
      ex_lowest_eigenvalue=list()
      for i in range (sample_size):
         h=myhaar_measureSO(N)
         ex.append(h)
         ex_lowest_eigenvalue.append(positivemin(h))
      with open(''+filename+'.csv', 'w') as archivo:
         for i in range(len(ex_lowest_eigenvalue)):
            for j in range(len(ex_lowest_eigenvalue[0])):
               archivo.write(str(ex_lowest_eigenvalue[i][j]))
               archivo.write("\n")
   else:
      return 'Not specified group'

#def generate_eigenangles_all_min(filename, N, sample_size, group_as_string, min):
#   if (group_as_string == 'symplectic'):
#      ex=list()
#      ex_lowest_eigenvalue=list()
#      for i in range (sample_size):
#         h=myhaar_measureUSp(N)
#         ex.append(h)
#         ex_lowest_eigenvalue.append(positivemin(h))
#      if min==False:
#         with open(''+filename+'.csv', 'w') as archivo:
#               for i in range(len(ex)):
#               for j in range(len(ex[0])):
#                  archivo.write(str(ex[i][j]))
#	          archivo.write("\n")
#      else:
#         with open(''+filename+'.csv', 'w') as archivo:
#            for i in range(len(ex_lowest_eigenvalue)):
#               for j in range(len(ex_lowest_eigenvalue[0])):
#                  archivo.write(str(ex_lowest_eigenvalue[i][j]))
#	          archivo.write("\n")
#   elif(group_as_string == 'unitary'):
#      ex=list()
#      ex_lowest_eigenvalue=list()
#      for i in range (sample_size):
#         h=myhaar_measureU(N)
#         ex.append(h)
#         ex_lowest_eigenvalue.append(positivemin(h))
#      if min==False:
#         with open(''+filename+'.csv', 'w') as archivo:
#               for i in range(len(ex)):
#               for j in range(len(ex[0])):
#                  archivo.write(str(ex[i][j]))
#	          archivo.write("\n")
#      else:
#         with open(''+filename+'.csv', 'w') as archivo:
#            for i in range(len(ex_lowest_eigenvalue)):
#               for j in range(len(ex_lowest_eigenvalue[0])):
#                  archivo.write(str(ex_lowest_eigenvalue[i][j]))
#	          archivo.write("\n")
#   elif(group_as_string == 'orthogonal'):
#      ex=list()
#      ex_lowest_eigenvalue=list()
#      for i in range (sample_size):
#         h=myhaar_measureSO(N)
#         ex.append(h)
#         ex_lowest_eigenvalue.append(positivemin(h))
#       if min==False:
#         with open(''+filename+'.csv', 'w') as archivo:
#               for i in range(len(ex)):
#               for j in range(len(ex[0])):
#                  archivo.write(str(ex[i][j]))
#	          archivo.write("\n")
#      else:
#         with open(''+filename+'.csv', 'w') as archivo:
#            for i in range(len(ex_lowest_eigenvalue)):
#               for j in range(len(ex_lowest_eigenvalue[0])):
#                  archivo.write(str(ex_lowest_eigenvalue[i][j]))
#	          archivo.write("\n")
#   else:
#      return 'Not specified group'
