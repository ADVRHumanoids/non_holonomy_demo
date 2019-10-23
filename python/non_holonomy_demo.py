#!/usr/bin/env python
from casadi import *
import matlogger2.matlogger as matl
import casadi_kin_dyn.pycasadi_kin_dyn as cas_kin_dyn
import rospy

logger = matl.MatLogger2('/tmp/non_holonomy_demo_log')
logger.setBufferMode(matl.BufferMode.CircularBuffer)

urdf = rospy.get_param('robot_description')
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

FK_base_link_str = kindyn.fk('base_link')
FK_base_link = Function.deserialize(FK_base_link_str)

id_string = kindyn.rnea()
ID = Function.deserialize(id_string)

tf = 2.  # Normalized time horizon
ns = 30  # number of shooting nodes

nc = 4  # number of contacts

DoF = 5
nq = 5
nv = 5

# Variables
q = SX.sym('q', nq)
qdot = SX.sym('qdot', nv)
qddot = SX.sym('qddot', nv)

# Bounds and initial guess

# CENTAURO homing
disp_z = 0.2

q_min = np.array([-1.0, -1.0, -100.0, # tx, tz, ry
                  -2.0, -2.0])        # wings

q_max = -q_min

q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

qdot_min = np.full((1, nv), -1000.)
qdot_max = np.full((1, nv), 1000.)
qdot_init = np.zeros_like(qdot_min)

qddot_min = np.full((1, nv), -1000.)
qddot_max = np.full((1, nv), 1000.)
qddot_init = np.zeros_like(qddot_min)

x_min = np.append(q_min, qdot_min)
x_max = np.append(q_max, qdot_max)

x_init = np.append(q_init, qdot_init)

x0_min = x_init
x0_max = x_init

qf_min = q_min
qf_max = q_max
qf_min[2] = 3.14
qf_max[2] = 3.14

xf_min = np.append(qf_min, np.zeros_like(qdot_min))
xf_max = np.append(qf_max, np.zeros_like(qdot_min))
#xf_min = x_init
#xf_max = x_init

t_min = 0.1
t_max = 0.1

# Model equations
x = vertcat(q, qdot)
xdot = vertcat(qdot, qddot)
nx = x.size1()

# Objective term
L = 0.01*dot(qddot, qddot)

# Runge-Kutta 4 integrator
f_RK = Function('f_RK', [x, qddot], [xdot, L])
X0 = MX.sym('X0', nx)
U = MX.sym('U', nv)
Time = MX.sym('Time', 1)
DT = Time
X = X0
Q = 0

k1, k1_q = f_RK(X, U)
k2, k2_q = f_RK(X + 0.5*DT*k1, U)
k3, k3_q = f_RK(X + DT / 2 * k2, U)
k4, k4_q = f_RK(X + DT * k3, U)
X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
Q = Q + DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)

F_RK = Function('F_RK', [X0, U, Time], [X, Q], ['x0', 'p', 'time'], ['xf', 'qf'])

# Start with an empty NLP
NV = nx*(ns+1) + nv*ns + ns
V = MX.sym('V', NV)

# NLP vars bounds and init guess
v_min = []
v_max = []
v_init = []
g_min = []
g_max = []

# offset in v
offset = 0

# "Lift" initial conditions
X = []
Qddot = []
Time = []

# Formulate the NLP
for k in range(ns):

    # State at k-th node
    X.append(V[offset:offset+nx])

    if k == 0:
        v_min += x0_min.tolist()
        v_max += x0_max.tolist()
    else:
        v_min += x_min.tolist()
        v_max += x_max.tolist()

    v_init += x_init.tolist()

    offset += nx

    # Control at k-th node
    Qddot.append(V[offset:offset+nv])

    v_min += qddot_min.tolist()
    v_max += qddot_max.tolist()

    v_init += qddot_init.tolist()

    offset += nv

    Time.append(V[offset:offset+1])
    v_min += np.array([t_min]).tolist()
    v_max += np.array([t_max]).tolist()
    v_init += np.array([t_min]).tolist()

    offset += 1

# Final state
X.append(V[offset:offset+nx])

v_min += xf_min.tolist()
v_max += xf_max.tolist()

v_init += x_init.tolist()

offset += nx

assert offset == NV

# Create NLP
J = MX([0])
g = []


tau_history = MX(Sparsity.dense(DoF, ns))
q_history = MX(Sparsity.dense(nq, ns+1))
qdot_history = MX(Sparsity.dense(nv, ns+1))
qddot_history = MX(Sparsity.dense(nv, ns))
h_history = MX(Sparsity.dense(1, ns))



for k in range(ns):

    integrator_out = F_RK(x0=X[k], p=Qddot[k], time=Time[k])

    Q_k = X[k][0:nq]
    Qdot_k = X[k][nq:nq + nv]

    q_zero = MX.zeros(nq)
    g_k = ID(q=Q_k, v=q_zero, a=q_zero)['tau']
    Tau_k = ID(q=Q_k, v=Qdot_k, a=Qddot[k])['tau'] - g_k

    J += 10*integrator_out['qf']
#    J += 1.*Time[k]
#    J += 1000.*dot(Qdot_k, Qdot_k)

    g += [integrator_out['xf'] - X[k+1]]
    g_min += [0]*X[k + 1].size1()
    g_max += [0]*X[k + 1].size1()

    g += [Tau_k[0:3]]
    g_min += np.zeros((3, 1)).tolist()
    g_max += np.zeros((3, 1)).tolist()

    # g += [Tau_k]
    # g_min += np.append(np.zeros((6, 1)), np.full((12, 1), -400.)).tolist()
    # g_max += np.append(np.zeros((6, 1)), np.full((12, 1), 400.)).tolist()

    tau_history[:, k] = Tau_k
    q_history[0:nq, k] = Q_k
    qdot_history[0:nv, k] = Qdot_k
    qddot_history[0:nv, k] = Qddot[k]
    h_history[0, k] = Time[k]

q_history[0:nq, ns] = X[ns][0:nq]
qdot_history[0:nv, ns] = X[ns][nq:nx]


g = vertcat(*g)
v_init = vertcat(*v_init)
g_min = vertcat(*g_min)
g_max = vertcat(*g_max)
v_min = vertcat(*v_min)
v_max = vertcat(*v_max)


# Create an NLP solver
prob = {'f': J, 'x': V, 'g': g}
opts = {'ipopt.tol': 1e-3,
        'ipopt.max_iter': 2000,
        'ipopt.linear_solver': 'ma57'}
solver = nlpsol('solver', 'ipopt', prob, opts)

# Solve the NLP
sol1 = solver(x0=v_init, lbx=v_min, ubx=v_max, lbg=g_min, ubg=g_max)
w_opt1 = sol1['x'].full().flatten()
lam_w_opt = sol1['lam_x']
lam_g_opt = sol1['lam_g']

# sol = solver(x0=w_opt1, lbx=v_min, ubx=v_max, lbg=g_min, ubg=g_max, lam_x0=lam_w_opt, lam_g0=lam_g_opt)
# w_opt = sol['x'].full().flatten()

w_opt = w_opt1

# Plot the solution
tau_hist = Function("tau_hist", [V], [tau_history])
tau_hist_value = tau_hist(w_opt).full()

q_hist = Function("q_hist", [V], [q_history])
q_hist_value = q_hist(w_opt).full()
qdot_hist = Function("qdot_hist", [V], [qdot_history])
qdot_hist_value = qdot_hist(w_opt).full()
qddot_hist = Function("qddot_hist", [V], [qddot_history])
qddot_hist_value = qddot_hist(w_opt).full()

h_hist = Function("h_hist", [V], [h_history])
h_hist_value = h_hist(w_opt).full()

logger.add('q', q_hist_value)
logger.add('qdot', qdot_hist_value)
logger.add('qddot', qddot_hist_value)
logger.add('tau', tau_hist_value)
logger.add('h', h_hist_value)
logger.add('ns', ns)


# Resampler
dt = 0.001

# Formulate discrete time dynamics
dae = {'x': x, 'p': qddot, 'ode': xdot, 'quad': []}
opts = {'tf': dt}
F_int = integrator('F_int', 'cvodes', dae, opts)

Tf = 0.0
T_i = {}

for k in range(ns):
    if k == 0:
        T_i[k] = 0.0
    else:
        T_i[k] = T_i[k-1] + h_hist_value[0, k-1]

    Tf += h_hist_value[0, k]

n_res = int(round(Tf/dt))

print(n_res)

q_res = MX(Sparsity.dense(nq, n_res))
qdot_res = MX(Sparsity.dense(nv, n_res))
qddot_res = MX(Sparsity.dense(nv, n_res))
X_res = MX(Sparsity.dense(nx, n_res+1))

k = 0

for i in range(ns):
    for j in range(int(round(h_hist_value[0, i]/dt))):

        n_prev = int(round(T_i[i]/dt))

        if j == 0:
            integrator_1 = F_int(x0=X[i], p=Qddot[i])
            X_res[0:nx, k+1] = integrator_1['xf']
        else:
            integrator_2 = F_int(x0=X_res[0:nx, k], p=Qddot[i])
            X_res[0:nx, k+1] = integrator_2['xf']

        q_res[0:nq, k] = X_res[0:nq, k+1]
        qdot_res[0:nv, k] = X_res[nq:nx, k+1]
        qddot_res[0:nv, k] = Qddot[i]

        k += 1

Resampler = Function("Resampler", [V], [q_res, qdot_res, qddot_res], ['V'], ['q_res', 'qdot_res', 'qddot_res'])

q_hist_res = Resampler(V=w_opt)['q_res'].full()
qdot_hist_res = Resampler(V=w_opt)['qdot_res'].full()
qddot_hist_res = Resampler(V=w_opt)['qddot_res'].full()

logger.add('q_resample', q_hist_res)
logger.add('qdot_resample', qdot_hist_res)
logger.add('qddot_resample', qddot_hist_res)


del(logger)

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import tf
import geometry_msgs.msg

pub = rospy.Publisher('joint_states', JointState, queue_size=10)
rospy.init_node('joint_state_publisher')
rate = rospy.Rate(1./dt)
joint_state_pub = JointState()
joint_state_pub.header = Header()
joint_state_pub.name = ['t_x_joint', 't_z_joint', 'rot_joint', 'left_joint', 'right_joint']


while not rospy.is_shutdown():
    rospy.sleep(1.0)
    for k in range(n_res):
        joint_state_pub.header.stamp = rospy.Time.now()
        joint_state_pub.position = q_hist_res[:, k]
        joint_state_pub.velocity = []
        joint_state_pub.effort = []
        pub.publish(joint_state_pub)
        rate.sleep()
