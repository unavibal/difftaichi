from math import isnan
from turtle import update
import taichi as ti
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras.api.optimizers as opt
import os


os.environ["KERAS_BACKEND"] = "tensorflow"

grad_needed = True
real = ti.f32
ti.init(arch=ti.cuda, default_fp=real, device_memory_GB=16)

dim = 2
width = 1.0
height = 0.025
N = 480  # reduce to 30 if run out of GPU memory
Nh = 10
n_particles = N * Nh
n_grid = 120
inner_grid_res = 10

dx = 1 / n_grid
inv_dx = 1 / dx
dt = 0.5e-4
p_mass = 10
p_vol = 1

E = ti.field(real, shape=(), needs_grad = grad_needed)
nu = ti.field(real, shape=(), needs_grad = grad_needed)

E[None] = 0.8e4
nu[None] = 0.4

mu = ti.field(real, shape=(), needs_grad = grad_needed)
la = ti.field(real, shape=(), needs_grad = grad_needed)

# mu = E[None]
# la = nu[None]

mu[None], la[None] = E[None] / (2.0 * (1.0 + nu[None])), E[None] * nu[None] / ((1.0 + nu[None]) * (1.0 - 2.0 * nu[None])) 

max_steps = 40024
steps = 40024
gravity = 9.8
target_area = 0.05
lower_bound = int(np.floor(n_particles*(0.5-target_area/2)))
upper_bound = int(np.ceil(n_particles*(0.5+target_area/2)))
target = [0.5, 0.4]

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

ti_target = ti.Vector.field(dim, shape=(1), dtype=real)
ti_target[0] = target

x = ti.Vector.field(dim,
                    dtype=real,
                    shape=(n_particles),
                    needs_grad=grad_needed)
new_x = ti.Vector.field(dim,
                    dtype=real,
                    shape=(n_particles),
                    needs_grad=grad_needed)

x_avg = ti.Vector.field(dim, dtype=real, shape=(), needs_grad=grad_needed)

v = ti.Vector.field(dim,
                    dtype=real,
                    shape=(n_particles),
                    needs_grad=grad_needed)
new_v = ti.Vector.field(dim,
                    dtype=real,
                    shape=(n_particles),
                    needs_grad=grad_needed)

grid_v_in = ti.Vector.field(dim,
                            dtype=real,
                            shape=(n_grid, n_grid),
                            needs_grad=grad_needed)


grid_v_out = ti.Vector.field(dim,
                             dtype=real,
                             shape=(n_grid, n_grid),
                             needs_grad=grad_needed)

grid_m_in = ti.field(dtype=real,
                     shape=(n_grid, n_grid),
                     needs_grad=grad_needed)


C = ti.Matrix.field(dim,
                    dim,
                    dtype=real,
                    shape=(n_particles),
                    needs_grad=grad_needed)
new_C = ti.Matrix.field(dim,
                    dim,
                    dtype=real,
                    shape=(n_particles),
                    needs_grad=grad_needed)

F = ti.Matrix.field(dim,
                    dim,
                    dtype=real,
                    shape=(n_particles),
                    needs_grad=grad_needed)
new_F = ti.Matrix.field(dim,
                    dim,
                    dtype=real,
                    shape=(n_particles),
                    needs_grad=grad_needed)

init_v = ti.Vector.field(dim, dtype=real, shape=(), needs_grad=grad_needed)
loss = ti.field(dtype=real, shape=(), needs_grad=grad_needed)


@ti.kernel
def set_v():
    for i in range(n_particles):
        v[i] = init_v[None]


@ti.kernel
def p2g(f: ti.i32):
    grid_m_in.fill(0.0)
    grid_v_in.fill(0.0)
    for p in range(n_particles):
        base = ti.cast(x[p] * inv_dx - 0.5, ti.i32)
        fx = x[p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F[p] = (ti.Matrix.diag(dim=2, val=1) + dt * C[p]) @ F[p]
        F[p] = new_F[p]
        J = (new_F[p]).determinant()
        r, s = ti.polar_decompose(new_F[p])
        cauchy = 2 * mu[None] * (new_F[p] - r) @ new_F[p].transpose() + \
                 ti.Matrix.diag(2, la[None] * (J - 1) * J)
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + p_mass * C[p]

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v_in[base + offset] += weight * (p_mass * v[p] +
                                                         affine @ dpos)
                grid_m_in[base + offset] += weight * p_mass


bound = 3
quad_damping_coef = ti.field(real, shape=(), needs_grad= grad_needed)
quad_damping_coef[None] = 8.0

@ti.kernel
def grid_op(f: ti.i32):
    for i, j in ti.ndrange(n_grid, n_grid):
        inv_m = 1 / (grid_m_in[i, j] + 1e-10)
        v_out = inv_m * grid_v_in[i, j]
        v_out[1] -= dt * gravity
        damping = ti.pow(v_out,2)*quad_damping_coef[None]
        v_out -= ti.math.sign(v_out) *damping*dt
        if i < bound:
            v_out[0] = 0
            v_out[1] = 0
        if i > n_grid - bound:
            v_out[0] = 0
            v_out[1] = 0
        if j < bound and v_out[1] < 0:
            v_out[1] = 0
        if j > n_grid - bound and v_out[1] > 0:
            v_out[1] = 0
        grid_v_out[i, j] = v_out


@ti.kernel
def g2p(f: ti.i32):
    # new_v.fill(0.0)
    # new_C.fill(0.0)
    
    for p in range(n_particles):
        base = ti.cast(x[p] * inv_dx - 0.5, ti.i32)
        fx = x[p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]

        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.cast(ti.Vector([i, j]), real) - fx
                g_v = grid_v_out[base[0] + i, base[1] + j]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        new_x[p] = x[p] + dt * new_v
        C[p]=new_C
        v[p]=new_v
    
    for p in range(n_particles):
        x[p]=new_x[p]



# @ti.kernel
# def compute_x_avg():
#     for i in range(n_particles):
#         x_avg[None] += (1 / n_particles) * x[steps - 1, i]

@ti.kernel
def update_x_avg_relevant():
    for i in range(lower_bound, upper_bound):
        x_avg[None] +=  (1 / (upper_bound-lower_bound)) * x[i]

@ti.kernel
def update_loss():
    for i in range(lower_bound, upper_bound):
        dist = (x[i]-target)**2
        loss[None] += 0.5*(1 / (upper_bound-lower_bound)) *(dist[0]+dist[1])

@ti.kernel
def comput_x_avg_relevant():
    for i in range(lower_bound, upper_bound):
        x_avg[None] +=  (1 / (upper_bound-lower_bound)) * x[i]


@ti.kernel
def compute_loss():
    dist = (x_avg[None] - ti.Vector(target))**2
    loss[None] = 0.5 * (dist[0] + dist[1])


def substep(s):
    p2g(s)
    grid_op(s)
    g2p(s)
    if s > 0.8*steps:
        update_loss()



x_width_base = (width-4/n_grid)/(N-1)
x_height_base = height/(Nh-1)

x_safety = 2/n_grid

inner_grid_n = inner_grid_res*inner_grid_res
y_safety=0
og_dx = (width-2*x_safety)/N
og_dy = (height-2*y_safety)/Nh

x_linspace = np.linspace(x_safety+og_dx/2, width-x_safety-og_dx/2, N)
y_linspace = np.linspace(y_safety+og_dy/2, height-y_safety-og_dy/2, Nh)

x_inner_linspace = np.linspace(-og_dx/2, og_dx/2, inner_grid_res)
y_inner_linspace = np.linspace(-og_dy/2, og_dy/2, inner_grid_res)

out_yy, out_xx = np.meshgrid(y_linspace, x_linspace)

in_xx, in_yy = np.meshgrid(x_inner_linspace, y_inner_linspace)
in_xx = np.reshape(in_xx, [-1,1]).squeeze(axis=1)
in_yy = np.reshape(in_yy, [-1,1]).squeeze(axis=1)




losses = []
img_count = 0

gui = ti.GUI("Simple Differentiable MPM Solver", (640, 640), 0xAAAAAA)
ui = ti.ui.Window("Simple Differentiable MPM Solver", (640, 640), fps_limit=1000)
canvas = ui.get_canvas()
canvas.set_background_color((0.0,0.0,0.0))

# n_frame = 150
# scale = 4
# for s in range(steps - 1):
#              substep(s) 
#              if s % n_frame==0:
#                 x_np =x.to_numpy()
#                 gui.circles(x_np[0:lower_bound, :], color=0x112233, radius=1.5)
#                 gui.circles(x_np[upper_bound:-1, :], color=0x112233, radius=1.5)
#                 gui.circles(x_np[lower_bound:upper_bound, :], color=0xFF0070, radius=1.5)
#                 gui.circle(target, radius=5, color=0xFFFFFF)
#                 img_count += 1
#                 gui.show()

n_frame=50
scale=4


mu_opt = tf.Variable([mu[None]])
la_opt = tf.Variable([la[None]])
damp_coef = tf.Variable([quad_damping_coef[None]])

mu_lr =  1.5*1e2
la_lr =   2.5*1e1
damp_lr = 2*1e-1

mu_min = 400.0
la_min = 1.0
damp_min = 0.1

mu_clip = 2000.0
la_clip = 2000.0
damp_clip = 20000.0

mu_adam = opt.Adam(learning_rate=mu_lr    )
la_adam= opt.Adam(learning_rate=la_lr     )
damp_adam = opt.Adam(learning_rate=damp_lr)

zero_grad = tf.constant([0])
mu_adam.apply_gradients(zip([zero_grad], [mu_opt]))
la_adam.apply_gradients(zip([zero_grad], [la_opt]))
damp_adam.apply_gradients(zip([zero_grad], [damp_coef]))


print("mu=", mu[None], "la=", la[None])

video_it = 1

for optim_i in range(100):

    init_v[None] = [0.0, 0.0]

    for i in range(n_particles):
        F[i] = [[1, 0], [0, 1]]
    
    for i in range(N):
        for j in range(Nh):
            # x[i * Nh + j] = [x_width_base*i+2/n_grid, x_height_base*j+0.5]
            # x[i*Nh+j] = np.random.rand(2)*np.array([width-4/n_grid, height])+np.array([2/n_grid,0.5])
            grid_choice = np.random.randint(0,inner_grid_n)
            x[i*Nh+j] = [out_xx[i,j]+in_xx[grid_choice], out_yy[i,j]+in_yy[grid_choice]+0.5]
        
    set_v()

    grid_v_in.fill(0)
    grid_m_in.fill(0)

    x_avg[None] = [0, 0] 
    loss[None] = 0.0
    with ti.ad.Tape(loss=loss):
        set_v()
        for s in range(steps - 1):
            substep(s)
            if s % n_frame==0 and optim_i % video_it==0:
                # x_np =x.to_numpy()
                # gui.circles(x_np[0:lower_bound, :], color=0x112233, radius=1.5)
                # gui.circles(x_np[upper_bound:-1, :], color=0x112233, radius=1.5)
                # gui.circles(x_np[lower_bound:upper_bound, :], color=0xFF0070, radius=1.5)
                # gui.circle(target, radius=5, color=0xFFFFFF)
                # img_count += 1
                # gui.show()
                canvas.set_background_color((0.7,0.7,0.7))
                canvas.circles(x, color=(0.3, 0.3, 0.3), radius=0.0015)
                # gui.circles(x_np[upper_bound:-1, :], color=0x112233, radius=1.5)
                # gui.circles(x_np[lower_bound:upper_bound, :], color=0xFF0070, radius=1.5)
                canvas.circles(ti_target, radius=0.005, color=(1, 1, 1))
                img_count += 1
                ui.show()

        # comput_x_avg_relevant()
        # compute_loss()

    l = loss[None]
    losses.append(l)

    mu_grad = mu.grad[None]
    la_grad = la.grad[None]
    damp_grad = quad_damping_coef.grad[None]

    print('loss=', l, '   grad=', (mu_grad, la_grad, damp_grad))

    if tf.norm(mu_grad) > mu_clip:
        mu_grad = np.sign(mu_grad)*mu_clip
    elif isnan(mu_grad):
        mu_grad = 0

    if tf.norm(la_grad) > la_clip:
        la_grad = np.sign(la_grad)*la_clip
    elif isnan(la_grad):
        la_grad = 0

    if tf.norm(damp_grad) > damp_clip:
        damp_grad = np.sign(damp_grad)* damp_clip
    elif isnan(damp_grad):
        damp_grad = 0

    mu_adam.apply([tf.constant([mu_grad])])
    la_adam.apply([tf.constant([la_grad])])
    damp_adam.apply([tf.constant([damp_grad])])

    mu[None] = mu_opt.numpy()[0]
    la[None] = la_opt.numpy()[0]

    if mu_opt.numpy()[0] > mu_min:
        mu[None] = mu_opt.numpy()[0]
    else: 
        mu_opt.assign([mu_min])
        mu_adam.set_weights([tf.constant(0.0), tf.constant(mu_lr), tf.constant([0.0]), tf.constant([0.0])])
        mu[None] = mu_min

    if la_opt.numpy()[0] > la_min:
        la[None] = la_opt.numpy()[0]
    else: 
        la_opt.assign([la_min])
        la_adam.set_weights([tf.constant(0.0), tf.constant(la_lr), tf.constant([0.0]), tf.constant([0.0])])
        la[None] = la_min

    if damp_coef.numpy()[0] > damp_min:
        quad_damping_coef[None] = damp_coef.numpy()[0]
    else: 
        damp_coef.assign([damp_min])
        damp_adam.set_weights([tf.constant(0.0), tf.constant(damp_lr), tf.constant([0.0]), tf.constant([0.0])])
        quad_damping_coef[None] = damp_min

    print("mu=", mu[None], "la=", la[None], "damp_coef=", quad_damping_coef[None])

plt.title("Optimization of Initial Velocity")
plt.ylabel("Loss")
plt.xlabel("Gradient Descent Iterations")
plt.plot(losses)
plt.show()
