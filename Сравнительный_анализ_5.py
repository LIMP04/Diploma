import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

alpha0 = np.radians([50, 30, 50, 40, 45, 30])
r0_K4 = position_K4(alpha0)

configs = [
    ('1. Pseudoinverse',     algo_pseudoinverse, {}),
    ('2. DLS',               algo_dls,           {'lam': 0.005}),
    ('3. CLIK',              algo_clik,          {'K': 30.0}),
    ('4. GPM',               algo_gpm,           {'k_null': 0.3}),
    ('5. ★ Гибрид',          algo_hybrid,        {'K': 30.0, 'lam': 0.005, 'k_null': 0.2}),
]

colors = ['#3498db', '#f39c12', '#27ae60', '#9b59b6', '#e74c3c']
lws = [2.0, 2.0, 2.0, 2.0, 3.0]

# ───── Тест 1: ПРЯМАЯ ─────
r_end1 = r0_K4 + np.array([0.030, 0.025, -0.020])
t1, r1, v1, _ = traj_line(r0_K4, r_end1, 0.02, 800)
res1 = [simulate(n, fn, r1, v1, t1, alpha0, **kw) for n, fn, kw in configs]

# ───── Тест 2: КРУГ ─────
c2 = r0_K4 + np.array([0.02, 0.02, -0.01])
t2, r2, v2, _ = traj_circle(c2, np.array([0.3, 0.2, 1.0]), 0.03, 0.5, 800)
alpha_c0 = alpha0.copy()
for _ in range(50):
    e = r2[0] - position_K4(alpha_c0)
    if np.linalg.norm(e) < 1e-7: break
    alpha_c0 += np.linalg.pinv(jacobian_K4(alpha_c0)) @ e
res2 = [simulate(n, fn, r2, v2, t2, alpha_c0, **kw) for n, fn, kw in configs]

# ───── Тест 3: КВАДРАТ ─────
n3 = np.array([0.2, 0.1, 1.0]); n3 /= np.linalg.norm(n3)
tmp = np.array([1, 0, 0])
e1p = tmp - np.dot(tmp, n3)*n3; e1p /= np.linalg.norm(e1p)
e2p = np.cross(n3, e1p)
sq_c = r0_K4 + 0.01*e1p + 0.01*e2p
t3, r3, v3, _ = traj_square(sq_c, e1p, e2p, 0.06, 0.025, 800)
alpha_s0 = alpha0.copy()
for _ in range(50):
    e = r3[0] - position_K4(alpha_s0)
    if np.linalg.norm(e) < 1e-7: break
    alpha_s0 += np.linalg.pinv(jacobian_K4(alpha_s0)) @ e
res3 = [simulate(n, fn, r3, v3, t3, alpha_s0, **kw) for n, fn, kw in configs]

# ───── Тест 4: СИНГУЛЯРНОСТЬ ─────
alpha_sing0 = np.radians([40, 30, 2, 30, 2, 30])
r0_sing = position_K4(alpha_sing0)
r_end4 = r0_sing + np.array([0.015, 0.010, 0.008])
t4, r4, v4, _ = traj_line(r0_sing, r_end4, 0.015, 800)
res4 = [simulate(n, fn, r4, v4, t4, alpha_sing0, **kw) for n, fn, kw in configs]

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'figure.facecolor': 'white', 'axes.facecolor': '#fafafa',
    'grid.color': '#cccccc', 'grid.linestyle': '--', 'grid.alpha': 0.7,
})

# ═══════════════════════════════════════════════════════════════════════
# РИС 1+: Прямая — С УВЕЛИЧЕННЫМИ ОТКЛОНЕНИЯМИ (масштаб 1:5000)
# ═══════════════════════════════════════════════════════════════════════
SCALE = 5000  

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

ax_left = fig.add_subplot(121, projection='3d')
ax_left.plot(r1[:,0], r1[:,1], r1[:,2], 'k--', linewidth=3,
             label='Требуемая r_d', zorder=10)
for r, c, lw in zip(res1, colors, lws):
    ax_left.plot(r['r_actual'][:,0], r['r_actual'][:,1], r['r_actual'][:,2],
                 color=c, linewidth=lw, label=r['name'])
ax_left.scatter(*r0_K4, color='#2ecc71', s=200, zorder=20, label='Старт')
ax_left.scatter(*r_end1, color='#e74c3c', s=200, marker='*', zorder=20, label='Финиш')
ax_left.set_title('РЕАЛЬНЫЙ масштаб (1:1)\nВсе алгоритмы выглядят одинаково',
                  fontweight='bold', fontsize=12)
ax_left.set_xlabel('X (м)'); ax_left.set_ylabel('Y (м)'); ax_left.set_zlabel('Z (м)')
ax_left.legend(fontsize=8, loc='upper left'); ax_left.view_init(elev=22, azim=55)
axes[0].axis('off')

# Справа: УВЕЛИЧЕННЫЕ отклонения
ax_right = fig.add_subplot(122, projection='3d')
ax_right.plot(r1[:,0], r1[:,1], r1[:,2], 'k--', linewidth=3,
              label='Требуемая r_d', zorder=10)
# Усиливаем отклонение от требуемой траектории в SCALE раз
for r, c, lw in zip(res1, colors, lws):
    r_amp = r1 + SCALE * (r['r_actual'] - r1)
    ax_right.plot(r_amp[:,0], r_amp[:,1], r_amp[:,2],
                  color=c, linewidth=lw, label=f"{r['name']}  ({r['max_err']*1e6:.2f} мкм)")
ax_right.scatter(*r0_K4, color='#2ecc71', s=200, zorder=20)
ax_right.scatter(*r_end1, color='#e74c3c', s=200, marker='*', zorder=20)
ax_right.set_title(f'УВЕЛИЧЕНИЕ ОТКЛОНЕНИЙ в {SCALE} раз\nТеперь видна разница алгоритмов',
                   fontweight='bold', fontsize=12, color='#c0392b')
ax_right.set_xlabel('X (м)'); ax_right.set_ylabel('Y (м)'); ax_right.set_zlabel('Z (м)')
ax_right.legend(fontsize=8, loc='upper left'); ax_right.view_init(elev=22, azim=55)
axes[1].axis('off')

fig.suptitle('Рис. 5.1*. Тест 1 — прямолинейная траектория K₄ (с увеличением отклонений)',
             fontsize=14, fontweight='bold', y=0.98)

SCALE2 = 80   

fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8))

ax_l = fig2.add_subplot(121, projection='3d')
ax_l.plot(r2[:,0], r2[:,1], r2[:,2], 'k--', linewidth=3, label='Требуемая (круг)', zorder=10)
for r, c, lw in zip(res2, colors, lws):
    ax_l.plot(r['r_actual'][:,0], r['r_actual'][:,1], r['r_actual'][:,2],
              color=c, linewidth=lw, label=r['name'])
ax_l.scatter(*r2[0], color='#2ecc71', s=200, zorder=20)
ax_l.set_title('РЕАЛЬНЫЙ масштаб (1:1)', fontweight='bold', fontsize=12)
ax_l.set_xlabel('X'); ax_l.set_ylabel('Y'); ax_l.set_zlabel('Z')
ax_l.legend(fontsize=8); ax_l.view_init(elev=20, azim=45)
axes2[0].axis('off')

ax_r = fig2.add_subplot(122, projection='3d')
ax_r.plot(r2[:,0], r2[:,1], r2[:,2], 'k--', linewidth=3, label='Требуемая (круг)', zorder=10)
for r, c, lw in zip(res2, colors, lws):
    r_amp = r2 + SCALE2 * (r['r_actual'] - r2)
    ax_r.plot(r_amp[:,0], r_amp[:,1], r_amp[:,2],
              color=c, linewidth=lw, label=f"{r['name']}  ({r['max_err']*1e6:.0f} мкм)")
ax_r.scatter(*r2[0], color='#2ecc71', s=200, zorder=20)
ax_r.set_title(f'УВЕЛИЧЕНИЕ в {SCALE2} раз\nCLIK и Гибрид — у требуемой,\nостальные — отклоняются',
               fontweight='bold', fontsize=12, color='#c0392b')
ax_r.set_xlabel('X'); ax_r.set_ylabel('Y'); ax_r.set_zlabel('Z')
ax_r.legend(fontsize=8); ax_r.view_init(elev=20, azim=45)
axes2[1].axis('off')

fig2.suptitle('Рис. 5.3*. Тест 2 — круговая траектория K₄ (с увеличением отклонений)',
              fontsize=14, fontweight='bold', y=0.98)

SCALE3 = 60

fig3, axes3 = plt.subplots(1, 2, figsize=(18, 8))

def to_plane(rr, c, e1, e2):
    d = rr - c
    return d @ e1, d @ e2

ax_l = axes3[0]
u_d, v_d = to_plane(r3, sq_c, e1p, e2p)
ax_l.plot(u_d, v_d, 'k--', linewidth=3, label='Требуемая (квадрат)', zorder=10)
for r, c, lw in zip(res3, colors, lws):
    u_a, v_a = to_plane(r['r_actual'], sq_c, e1p, e2p)
    ax_l.plot(u_a, v_a, color=c, linewidth=lw, label=r['name'])
ax_l.set_xlabel('u (м)'); ax_l.set_ylabel('v (м)')
ax_l.set_title('РЕАЛЬНЫЙ масштаб (1:1)', fontweight='bold', fontsize=12)
ax_l.legend(fontsize=8); ax_l.grid(True); ax_l.set_aspect('equal')

ax_r = axes3[1]
ax_r.plot(u_d, v_d, 'k--', linewidth=3, label='Требуемая (квадрат)', zorder=10)
for r, c, lw in zip(res3, colors, lws):
    u_a, v_a = to_plane(r['r_actual'], sq_c, e1p, e2p)
    u_amp = u_d + SCALE3 * (u_a - u_d)
    v_amp = v_d + SCALE3 * (v_a - v_d)
    ax_r.plot(u_amp, v_amp, color=c, linewidth=lw,
              label=f"{r['name']}  ({r['max_err']*1e6:.0f} мкм)")
ax_r.set_xlabel('u (м)'); ax_r.set_ylabel('v (м)')
ax_r.set_title(f'УВЕЛИЧЕНИЕ в {SCALE3} раз\nна углах квадрата ошибки максимальны',
               fontweight='bold', fontsize=12, color='#c0392b')
ax_r.legend(fontsize=8); ax_r.grid(True); ax_r.set_aspect('equal')

fig3.suptitle('Рис. 5.4*. Тест 3 — квадрат в плоскости (проекция, с увеличением)',
              fontsize=14, fontweight='bold', y=0.98)
fig3.tight_layout()


fig4, axes4 = plt.subplots(2, 2, figsize=(16, 10))
test_data = [(res1, 'Тест 1: Прямая', axes4[0,0]),
             (res2, 'Тест 2: Круг', axes4[0,1]),
             (res3, 'Тест 3: Квадрат', axes4[1,0]),
             (res4, 'Тест 4: Сингулярность', axes4[1,1])]

for results, title, ax in test_data:
    for r, c, lw in zip(results, colors, lws):
        ax.semilogy(r['t'], r['err']*1e6 + 1e-3,
                    color=c, linewidth=lw, label=r['name'])
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlabel('Время t (с)')
    ax.set_ylabel('Ошибка |Δr| (мкм, лог. масштаб)')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, which='both', alpha=0.5)

fig4.suptitle('Рис. 5.2*. Ошибка следования во времени — все 4 теста\n(логарифмическая шкала наглядно показывает разницу)',
              fontsize=14, fontweight='bold')
fig4.tight_layout()



fig5, axes5 = plt.subplots(2, 3, figsize=(18, 10))
axes5 = axes5.flatten()
for idx, r in enumerate(res4):
    ax = axes5[idx]
    for j in range(6):
        ax.plot(r['t'], r['alpha_dot'][:, j], linewidth=1.5, label=f'α̇{j+1}')
    ax.set_title(r['name'], fontweight='bold', fontsize=12)
    ax.set_xlabel('t (с)'); ax.set_ylabel('α̇ (рад/с)')
    ax.grid(True); ax.legend(fontsize=8, ncol=3)

    # Добавляем линию физического предела привода (10 рад/с)
    ax.axhline(y=10, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=-10, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.5)

    if r['max_alpha_dot'] > 5:
        ax.set_facecolor('#fadbd8')
        ax.text(0.5, 0.95, f'⚠ ВЗРЫВ: max|α̇| = {r["max_alpha_dot"]:.0f} рад/с',
                transform=ax.transAxes, ha='center', fontsize=12,
                fontweight='bold', color='#c0392b',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#c0392b'))
    else:
        ax.set_facecolor('#eafaf1')
        ax.text(0.5, 0.95, f'✓ ОК: max|α̇| = {r["max_alpha_dot"]:.2f} рад/с',
                transform=ax.transAxes, ha='center', fontsize=12,
                fontweight='bold', color='#27ae60',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#27ae60'))

axes5[-1].axis('off')
axes5[-1].text(0.5, 0.5,
               'ТЕСТ 4: Сингулярность\n\n'
               'α₃ ≈ 0°, α₅ ≈ 0°\n'
               '(манипулятор почти выпрямлен)\n\n'
               '─────────────\n\n'
               'w(α₀) = 0,011 (мала́)\n'
               'cond(J) = 87 (высока́я)\n\n'
               '─────────────\n\n'
               'Только DLS и Гибрид\n'
               'остаются в пределах\n'
               'физических возможностей\n'
               'привода (~10 рад/с,\n'
               'красная пунктирная линия)',
               ha='center', va='center', fontsize=11,
               bbox=dict(boxstyle='round,pad=1', facecolor='#fff9e6',
                        edgecolor='#bba60d', linewidth=2))
fig5.suptitle('Рис. 5.5*. α̇(t) вблизи сингулярности — наглядное сравнение',
              fontsize=14, fontweight='bold')
fig5.tight_layout()


fig6, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 7))

names = ['Pseudo-\ninverse', 'DLS', 'CLIK', 'GPM', '★ Гибрид']
errs1 = [r['max_err']*1e6 for r in res1]
errs2 = [r['max_err']*1e6 for r in res2]
errs3 = [r['max_err']*1e6 for r in res3]
rates4 = [r['max_alpha_dot'] for r in res4]


x = np.arange(len(names))
w = 0.27
ax_l.bar(x - w, errs1, w, color='#3498db', label='Тест 1: Прямая')
ax_l.bar(x, errs2, w, color='#27ae60', label='Тест 2: Круг')
ax_l.bar(x + w, errs3, w, color='#f39c12', label='Тест 3: Квадрат')
ax_l.set_yscale('log')
ax_l.set_xticks(x); ax_l.set_xticklabels(names, fontsize=10)
ax_l.set_ylabel('Δ_max, мкм (лог. шкала)', fontsize=11)
ax_l.set_title('Точность следования: Δ_max в трёх тестах', fontweight='bold', fontsize=12)
ax_l.legend(fontsize=10); ax_l.grid(True, axis='y', which='both', alpha=0.4)


for i in range(len(names)):
    ax_l.text(i - w, errs1[i]*1.3, f'{errs1[i]:.1f}', ha='center', fontsize=8)
    ax_l.text(i, errs2[i]*1.3, f'{errs2[i]:.0f}', ha='center', fontsize=8)
    ax_l.text(i + w, errs3[i]*1.3, f'{errs3[i]:.0f}', ha='center', fontsize=8)


bar_colors = ['#e74c3c' if r > 10 else '#27ae60' for r in rates4]
bars = ax_r.bar(x, rates4, color=bar_colors, edgecolor='black', linewidth=0.5)
ax_r.set_yscale('log')
ax_r.axhline(y=10, color='#c0392b', linestyle='--', linewidth=2,
             label='Физический предел привода (~10 рад/с)')
ax_r.set_xticks(x); ax_r.set_xticklabels(names, fontsize=10)
ax_r.set_ylabel('max|α̇|, рад/с (лог. шкала)', fontsize=11)
ax_r.set_title('Тест 4 (СИНГУЛЯРНОСТЬ): max|α̇|',
               fontweight='bold', fontsize=12, color='#c0392b')
ax_r.legend(fontsize=10, loc='lower right'); ax_r.grid(True, axis='y', which='both', alpha=0.4)

for i, (bar, rate) in enumerate(zip(bars, rates4)):
    h = bar.get_height()
    status = '⚠ ВЗРЫВ' if rate > 10 else '✓ OK'
    color = '#c0392b' if rate > 10 else '#27ae60'
    ax_r.text(bar.get_x() + bar.get_width()/2, h*1.5,
              f'{rate:.1f}\n{status}', ha='center', fontsize=9,
              fontweight='bold', color=color)

fig6.suptitle('Рис. 5.6*. Сводное сравнение алгоритмов — столбиковые диаграммы',
              fontsize=14, fontweight='bold', y=0.98)
fig6.tight_layout()
