Included with the problem set are 2 pickle files, each defining sequences of control inputs and point feature observations recorded as the robot navigated using different motion and measurement noise.


    U:   A 3 x T array of control inputs, where each column is of the form
         [t; d; dtheta] and specifies the forward distance (d) and change
         in heading (dtheta) applied at time step t

    Z:   4 x n array of observations, where each column is of the form
         [t; id; x; y] corresponds to a measurement of the relative position
         of landmark id acquired at time step t

    R:   Motion model noise covariance matrix

    Q:   Measurement model noise covariance matrix

    X0:  A 3 element vector specifying the initial pose (x, y, theta)

    XGT: A 4 x (T+1) array specifying the ground-truth pose, where each
         column is of the form [t; x; y; theta] and the first column is
         the initial pose

    MGT: 3 x M array specifying the map, where each column is of the form
         [id; x; y] and indicates the (x,y) position of landmark with given id
