import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

L1,L2,L3 = 0.50, 0.40, 0.30
l1,l2,l3 = 0.25, 0.20, 0.15

a1 = np.radians(60)   # α₁ — полярный K1
a2 = np.radians(45)   # α₂ — азимут K1
a3 = np.radians(30)   # α₃ — добавка полярного → K2
a4 = np.radians(30)   # α₄ — азимут K2
a5 = np.radians(20)   # α₅ — добавка полярного → K3
a6 = np.radians(20)   # α₆ — азимут K3

da1,da2,da3,da4,da5,da6 = 1.0, 0.5, 0.3, 0.4, 0.2, 0.3
dda1,dda2 = 0.10, 0.05

print("="*65)
print("  СРАВНЕНИЕ: ОБОБЩЁННЫЕ vs СФЕРИЧЕСКИЕ КООРДИНАТЫ")
print("  Полная модель (6 углов)")
print("="*65)
print(f"""
Частный случай:
  α₁={np.degrees(a1):.1f}° (полярный K1)   α₂={np.degrees(a2):.1f}° (азимут K1)
  α₃={np.degrees(a3):.1f}° (добавка → K2)  α₄={np.degrees(a4):.1f}° (азимут K2)
  α₅={np.degrees(a5):.1f}° (добавка → K3)  α₆={np.degrees(a6):.1f}° (азимут K3)
  α̇₁={da1}, α̇₂={da2}, α̇₃={da3}, α̇₄={da4}, α̇₅={da5}, α̇₆={da6} рад/с
""")

# K1: q1=l1·cos(α2)·sin(α1), q2=l1·sin(α1)·sin(α2), q3=l1·cos(α1)
K1 = np.array([
    l1*np.cos(a2)*np.sin(a1),
    l1*np.sin(a1)*np.sin(a2),
    l1*np.cos(a1)
])

# O1 — конец 1-го звена
O1 = np.array([
    L1*np.sin(a1)*np.cos(a2),
    L1*np.sin(a1)*np.sin(a2),
    L1*np.cos(a1)
])

# K2: q4=L1·sin(α1)·cos(α2)+l2·sin(α1+α3)·cos(α4), ...
K2 = np.array([
    L1*np.sin(a1)*np.cos(a2) + l2*np.sin(a1+a3)*np.cos(a4),
    L1*np.sin(a1)*np.sin(a2) + l2*np.sin(a1+a3)*np.sin(a4),
    L1*np.cos(a1)            + l2*np.cos(a1+a3)
])

# O2 — конец 2-го звена
O2 = np.array([
    L1*np.sin(a1)*np.cos(a2) + L2*np.sin(a1+a3)*np.cos(a4),
    L1*np.sin(a1)*np.sin(a2) + L2*np.sin(a1+a3)*np.sin(a4),
    L1*np.cos(a1)            + L2*np.cos(a1+a3)
])

# K3: q7=L1·cos(α2)·sin(α1)+L2·sin(α1+α3)·cos(α4)+l3·sin(α1+α3+α5)·cos(α6), ...
K3 = np.array([
    L1*np.cos(a2)*np.sin(a1) + L2*np.sin(a1+a3)*np.cos(a4) + l3*np.sin(a1+a3+a5)*np.cos(a6),
    L1*np.sin(a2)*np.sin(a1) + L2*np.sin(a1+a3)*np.sin(a4) + l3*np.sin(a1+a3+a5)*np.sin(a6),
    L1*np.cos(a1)            + L2*np.cos(a1+a3)            + l3*np.cos(a1+a3+a5)
])


def sph(p):
    r  = np.linalg.norm(p)
    th = np.degrees(np.arccos(np.clip(p[2]/r,-1,1)))
    ph = np.degrees(np.arctan2(p[1],p[0]))
    return r, th, ph

r1,th1,ph1 = sph(K1)
r2,th2,ph2 = sph(K2)
r3,th3,ph3 = sph(K3)

print("─"*65)
print("  РАЗДЕЛ 1: ДЕКАРТОВЫ КООРДИНАТЫ (формула 15)")
print("─"*65)
for nm,K,r,th,ph in [('K₁',K1,r1,th1,ph1),('K₂',K2,r2,th2,ph2),('K₃',K3,r3,th3,ph3)]:
    print(f"  {nm}: x={K[0]:.4f}, y={K[1]:.4f}, z={K[2]:.4f}  | r={r:.4f}м, θ={th:.2f}°, φ={ph:.2f}°")


print("\n" + "─"*65)
print("  РАЗДЕЛ 2: ПРЯМОЕ СРАВНЕНИЕ УГЛОВ")
print("─"*65)

# Обобщённые полярные (накапливаются по цепи)
gen_polar = [np.degrees(a1), np.degrees(a1+a3), np.degrees(a1+a3+a5)]
gen_azim  = [np.degrees(a2), np.degrees(a4),    np.degrees(a6)]
sph_polar = [th1, th2, th3]
sph_azim  = [ph1, ph2, ph3]

print(f"""
  ┌──────┬──────────────────────────┬───────────────────────────┐
  │Точка │  Полярный угол           │  Азимутальный угол        │
  │      │  Обобщ. (Σαᵢ)│Сфер.(θ)  │  Обобщ.(αᵢ) │ Сфер.(φ)  │
  ├──────┼─────────────────────────┼───────────────────────────┤
  │  K₁  │  α₁={gen_polar[0]:5.1f}°  │ θ={sph_polar[0]:5.2f}°│  α₂={gen_azim[0]:5.1f}°  │ φ={sph_azim[0]:6.2f}° │
  │  K₂  │α₁+α₃={gen_polar[1]:5.1f}° │ θ={sph_polar[1]:5.2f}°│  α₄={gen_azim[1]:5.1f}°  │ φ={sph_azim[1]:6.2f}° │
  │  K₃  │α₁+α₃+α₅={gen_polar[2]:3.1f}°│ θ={sph_polar[2]:5.2f}°│  α₆={gen_azim[2]:5.1f}°  │ φ={sph_azim[2]:6.2f}° │
  └──────┴─────────────────────────┴───────────────────────────┘

  Разности полярных:
    K₁: |α₁ − θ|     = {abs(gen_polar[0]-sph_polar[0]):.4f}°  {'← НУЛЬ ✓' if abs(gen_polar[0]-sph_polar[0])<0.01 else '← не совпадают'}
    K₂: |α₁+α₃ − θ|  = {abs(gen_polar[1]-sph_polar[1]):.4f}°  ← не совпадают
    K₃: |Σαᵢ − θ|    = {abs(gen_polar[2]-sph_polar[2]):.4f}°  ← не совпадают
""")



# ════════════════════════════════════════════════════════════════════
#  ВИЗУАЛИЗАЦИЯ — 5 графиков
# ════════════════════════════════════════════════════════════════════
DARK='#07090f'; SURF='#0f1320'; GRID='#1e2535'
TEXT='#dde4f0'; MUTED='#5a6480'
BLUE='#4d9fff'; GREEN='#3ecf72'; ORANGE='#f5873a'
YELLOW='#f0c040'; PURPLE='#a87aff'; CYAN='#38c8d0'; RED='#f06060'

fig = plt.figure(figsize=(20,14))
fig.patch.set_facecolor(DARK)
gs = GridSpec(2,3,figure=fig,hspace=0.42,wspace=0.32,
              left=0.05,right=0.97,top=0.91,bottom=0.06)

# ── График 1: 3D манипулятор ─────────────────────────────────────
ax1 = fig.add_subplot(gs[:,0],projection='3d')
ax1.set_facecolor(DARK)
for p in [ax1.xaxis.pane,ax1.yaxis.pane,ax1.zaxis.pane]:
    p.set_facecolor(SURF); p.set_edgecolor(GRID)

O0=np.array([0,0,0])
END=np.array([
    L1*np.sin(a1)*np.cos(a2)+L2*np.sin(a1+a3)*np.cos(a4)+L3*np.sin(a1+a3+a5)*np.cos(a6),
    L1*np.sin(a1)*np.sin(a2)+L2*np.sin(a1+a3)*np.sin(a4)+L3*np.sin(a1+a3+a5)*np.sin(a6),
    L1*np.cos(a1)+L2*np.cos(a1+a3)+L3*np.cos(a1+a3+a5)
])

# Звенья
for A,B,col,lw in [(O0,O1,BLUE,5),(O1,O2,GREEN,4),(O2,END,ORANGE,3.5)]:
    ax1.plot(*zip(A,B),color=col,lw=lw)
    ax1.plot(*zip(A,B),color=col,lw=lw*3,alpha=0.1)

# Суставы
for pt,nm in [(O0,'O₀'),(O1,'O₁'),(O2,'O₂')]:
    ax1.scatter(*pt,color=RED,s=80,zorder=5)
    ax1.text(pt[0]+.02,pt[1]+.02,pt[2]+.02,nm,color=RED,fontsize=9,fontweight='bold')

# Центры масс
for pt,nm,col in [(K1,'K₁',YELLOW),(K2,'K₂',YELLOW),(K3,'K₃',YELLOW)]:
    ax1.scatter(*pt,color=col,s=80,zorder=5,marker='D')
    ax1.text(pt[0]+.02,pt[1]+.02,pt[2]+.02,nm,color=col,fontsize=9,fontweight='bold')

# Сфера K1
u=np.linspace(0,2*np.pi,30); v=np.linspace(0,np.pi,20)
sx=l1*np.outer(np.cos(u),np.sin(v))
sy=l1*np.outer(np.sin(u),np.sin(v))
sz=l1*np.outer(np.ones(30),np.cos(v))
ax1.plot_surface(sx,sy,sz,alpha=0.06,color=BLUE)

# Проекции
for pt in [K1,K2,K3]:
    ax1.plot([pt[0],pt[0]],[pt[1],pt[1]],[0,pt[2]],'--',color=YELLOW,lw=0.7,alpha=0.4)

ax1.set_xlabel('X',color=TEXT,fontsize=9); ax1.set_ylabel('Y',color=TEXT,fontsize=9)
ax1.set_zlabel('Z',color=TEXT,fontsize=9)
ax1.tick_params(colors=TEXT,labelsize=7)
ax1.set_title('3D схема манипулятора\n(полная модель, формула 19)',color=TEXT,fontsize=10)
ax1.view_init(elev=20,azim=42)

from matplotlib.lines import Line2D
leg=[Line2D([0],[0],color=BLUE,lw=2,label='Звено 1 (α₁,α₂)'),
     Line2D([0],[0],color=GREEN,lw=2,label='Звено 2 (α₃,α₄)'),
     Line2D([0],[0],color=ORANGE,lw=2,label='Звено 3 (α₅,α₆)'),
     Line2D([0],[0],color=YELLOW,lw=0,marker='D',markersize=5,label='K₁,K₂,K₃'),
     Line2D([0],[0],color=BLUE,lw=0.7,alpha=0.4,linestyle='--',label='Сфера r=l₁')]
ax1.legend(handles=leg,facecolor=SURF,edgecolor=GRID,labelcolor=TEXT,fontsize=8,loc='upper left')

# ── График 2: Сравнение полярных углов ───────────────────────────
ax2 = fig.add_subplot(gs[0,1])
ax2.set_facecolor(DARK)

x=np.arange(3); w=0.32
b1=ax2.bar(x-w/2,gen_polar,w,label='Обобщённые Σαᵢ',color=BLUE,alpha=0.85)
b2=ax2.bar(x+w/2,sph_polar,w,label='Сферические θ', color=ORANGE,alpha=0.85)

for bar,v in zip(b1,gen_polar):
    ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,
             f'{v:.1f}°',ha='center',va='bottom',color=TEXT,fontsize=8)
for bar,v in zip(b2,sph_polar):
    ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,
             f'{v:.1f}°',ha='center',va='bottom',color=TEXT,fontsize=8)

ax2.set_xticks(x); ax2.set_xticklabels(['K₁','K₂','K₃'],color=TEXT,fontsize=11)
ax2.set_ylabel('Полярный угол, °',color=TEXT,fontsize=9)
ax2.set_title('Полярный угол\nОбобщённые Σαᵢ  vs  Сферические θ',color=TEXT,fontsize=10)
ax2.legend(facecolor=SURF,edgecolor=GRID,labelcolor=TEXT,fontsize=8)
ax2.tick_params(colors=TEXT,labelsize=8)
for sp in ax2.spines.values(): sp.set_color(GRID)

ax2.annotate('СОВПАДАЮТ\n✓',xy=(0,max(gen_polar[0],sph_polar[0])+1),
             ha='center',color=GREEN,fontsize=8,fontweight='bold')
ax2.annotate('≠',xy=(1,max(gen_polar[1],sph_polar[1])+3),
             ha='center',color=RED,fontsize=20,fontweight='bold')
ax2.annotate('≠',xy=(2,max(gen_polar[2],sph_polar[2])+3),
             ha='center',color=RED,fontsize=20,fontweight='bold')

# ── График 3: Сравнение азимутов ─────────────────────────────────
ax3 = fig.add_subplot(gs[0,2])
ax3.set_facecolor(DARK)

b3=ax3.bar(x-w/2,gen_azim,w,label='Обобщённые (α₂,α₄,α₆)',color=PURPLE,alpha=0.85)
b4=ax3.bar(x+w/2,sph_azim,w,label='Сферические φ',         color=CYAN,alpha=0.85)

for bar,v in zip(b3,gen_azim):
    ax3.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,
             f'{v:.1f}°',ha='center',va='bottom',color=TEXT,fontsize=8)
for bar,v in zip(b4,sph_azim):
    ax3.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,
             f'{v:.1f}°',ha='center',va='bottom',color=TEXT,fontsize=8)

ax3.set_xticks(x); ax3.set_xticklabels(['K₁','K₂','K₃'],color=TEXT,fontsize=11)
ax3.set_ylabel('Азимут, °',color=TEXT,fontsize=9)
ax3.set_title('Азимутальный угол\nОбобщённые (α₂,α₄,α₆)  vs  Сферические φ',color=TEXT,fontsize=10)
ax3.legend(facecolor=SURF,edgecolor=GRID,labelcolor=TEXT,fontsize=8)
ax3.tick_params(colors=TEXT,labelsize=8)
for sp in ax3.spines.values(): sp.set_color(GRID)

# Аннотации
for i,(gv,sv) in enumerate(zip(gen_azim,sph_azim)):
    match = abs(gv-sv)<0.01
    ax3.annotate('≡ ✓' if match else '≠',
                 xy=(i,max(gv,sv)+1),ha='center',
                 color=GREEN if match else RED,fontsize=9,fontweight='bold')

# ── График 4: Радиус r ────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1,1])
ax4.set_facecolor(DARK)

rs = [r1,r2,r3]; consts=[l1,None,None]
colors=[BLUE,GREEN,ORANGE]; nms=['K₁\n(r=l₁=const)','K₂\n(r меняется)','K₃\n(r меняется)']
bars=ax4.bar(nms,rs,color=colors,alpha=0.85,width=0.5)
ax4.axhline(l1,color=BLUE,lw=1.5,linestyle='--',alpha=0.7,label=f'l₁={l1} м')

for bar,v in zip(bars,rs):
    ax4.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
             f'{v:.4f} м',ha='center',va='bottom',color=TEXT,fontsize=9,fontweight='bold')

ax4.set_ylabel('r, м',color=TEXT,fontsize=9)
ax4.set_title('Радиус r от начала координат\nK₁: r=l₁=const (сфера)',color=TEXT,fontsize=10)
ax4.legend(facecolor=SURF,edgecolor=GRID,labelcolor=TEXT,fontsize=8)
ax4.tick_params(colors=TEXT,labelsize=8)
for sp in ax4.spines.values(): sp.set_color(GRID)

ax4.text(0,l1/2,'r = l₁ ✓',ha='center',color=BLUE,fontsize=9,fontweight='bold')


# Итоговые данные 
results = {
    'K1':K1,'K2':K2,'K3':K3,
    'r1':r1,'r2':r2,'r3':r3,
    'th1':th1,'th2':th2,'th3':th3,
    'ph1':ph1,'ph2':ph2,'ph3':ph3,
    'gen_polar':gen_polar,'gen_azim':gen_azim,
    'sph_polar':sph_polar,'sph_azim':sph_azim,
}
