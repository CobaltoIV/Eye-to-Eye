n_x = 8
n_y = 5
x_i = 44
y_i = 44
l = 30

dummy = "{c_x:.4f} {c_y:.4f} {c_z:.4f} \n"
with open('model.txt','w') as output:
    for y in range(n_y):
        for x in range(n_x):
            curr_x = (n_x-1)*l + x_i - x*l
            curr_y = y_i + y*l
            output.write(dummy.format( c_x = curr_x, c_y = curr_y, c_z=0 ))
