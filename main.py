"""Live procedural rose animation. Run: python main.py"""
from __future__ import annotations
import math, random
import numpy as np
import pygame
import pygame.gfxdraw

W, H = 1080, 1920
BLACK = (0, 0, 0)


def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def smooth(t):
    t = clamp(t); return t*t*(3-2*t)
def ease_out(t):
    t = clamp(t); return 1-(1-t)**3

def cubic(a, b, c, d, t):
    q = 1-t
    return q*q*q*a + 3*q*q*t*b + 3*q*t*t*c + t*t*t*d


def glow_surface(radius: int, color, strength=1.0):
    size = radius*2+1
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    d = np.sqrt(xx*xx+yy*yy)/max(1,radius)
    intensity = np.exp(-d*d*5)*strength
    a = (intensity*90).clip(0,255).astype(np.uint8)
    arr = np.zeros((size,size,4), dtype=np.uint8)
    arr[:,:,:3] = (intensity[:,:,None]*np.array(color)).clip(0,255).astype(np.uint8)
    arr[:,:,3] = a
    pygame.surfarray.blit_array(s, arr[:,:,:3].transpose(1,0,2))
    # Alpha is assigned through pixels_alpha (pygame's x/y convention).
    pygame.surfarray.pixels_alpha(s)[:] = a.T
    return s


class Petal:
    """Procedural cupped petal drawn as a shaded, softly curved polygon."""
    def __init__(self, layer, index, total, seed):
        self.layer=layer; self.index=index; self.total=total; self.seed=seed
        self.angle=math.tau*index/total + layer*.19
        self.scale=random.uniform(.88,1.12)
        self.width=random.uniform(.78,1.12)
        self.color=random.choice([(116,3,25),(151,5,34),(184,10,45),(215,24,57),(236,54,75),(170,12,42)])
        self.phase=random.random()*math.tau

    def polygon(self, cx, cy, t, time):
        bloom=smooth((t-self.layer*.105)/.72)
        r=(30+self.layer*33)*self.scale*bloom
        length=(100+self.layer*23)*self.scale*bloom
        spread=(26+self.layer*4)*self.width*bloom
        sway=math.sin(time*.55+self.phase)*.012
        a=self.angle+sway
        tip=(math.sin(time*.7+self.phase)*.035)
        left=((0,0),(-spread*.2,-length*.1),(-spread*1.15,-length*.48),
              (-spread*.72,-length*.83))
        right=((-spread*.72,-length*.83),(-spread*.62,-length*(1+tip)),
               (spread*.62,-length*(1+tip)),(spread*.72,-length*.83))
        back=((spread*.72,-length*.83),(spread*1.15,-length*.48),
              (spread*.2,-length*.1),(0,0))
        local=[]
        for curve in (left,right,back):
            for i in range(9):
                q=i/8
                local.append((cubic(*(point[0] for point in curve),q),
                              cubic(*(point[1] for point in curve),q)))
        ca,sa=math.cos(a),math.sin(a)
        pts=[]
        for lx,ly in local:
            px=lx*ca-ly*sa; py=lx*sa+ly*ca
            pts.append((cx+px+r*math.cos(a), cy+py+r*.25+r*math.sin(a)))
        return pts, bloom

    def draw(self, surface, cx, cy, t, time):
        pts,bloom=self.polygon(cx,cy,t,time)
        if bloom<=0: return
        base=self.color
        shadow=[(x+7,y+12) for x,y in pts]
        pixel_shadow=[(round(x),round(y)) for x,y in shadow]
        pixel_pts=[(round(x),round(y)) for x,y in pts]
        pygame.gfxdraw.filled_polygon(surface,pixel_shadow,(22,0,8))
        pygame.gfxdraw.filled_polygon(surface,pixel_pts,base)
        pygame.gfxdraw.aapolygon(surface,pixel_pts,base)
        light_strength=.92+.08*math.sin(time*.8+self.phase)
        light=tuple(min(255,int(c*(1.18*light_strength)+10)) for c in base)
        deep=tuple(int(c*.58) for c in base)
        pygame.draw.aalines(surface,deep,False,pixel_pts[:9])
        pygame.draw.aalines(surface,light,False,pixel_pts[10:17])
        if self.layer < 3:
            mid=[pixel_pts[i] for i in (0,4,8,13,17,22,26)]
            pygame.draw.aalines(surface,tuple(int(c*.82) for c in base),False,mid)


class Leaf:
    def __init__(self, side, height, size): self.side=side; self.height=height; self.size=size
    def draw(self, surface, cx, stem_top, progress, time):
        p=smooth((progress-.36)/.42)
        if p<=0:return
        x0=cx; y0=stem_top+self.height
        x1=x0+self.side*self.size*p; y1=y0-self.size*.30*p
        perp=(-self.side*.25, -.97)
        pts=[(x0,y0),(x0+self.side*self.size*.24*p,y0-self.size*.20*p),
             (x1+perp[0]*self.size*.28*p,y1+perp[1]*self.size*.28*p),
             (x1,y1),(x1-perp[0]*self.size*.28*p,y1-perp[1]*self.size*.28*p),
             (x0-self.side*self.size*.08*p,y0+self.size*.05*p)]
        pygame.draw.polygon(surface,(8,62,34),pts)
        pygame.draw.lines(surface,(18,83,43),True,pts,2)
        pygame.draw.line(surface,(143,165,79),(x0,y0),(x1,y1),2)
        for k in range(1,8):
            q=k/9; bx=x0+(x1-x0)*q; by=y0+(y1-y0)*q
            reach=self.size*.13*p*math.sin(math.pi*q)
            pygame.draw.line(surface,(71,132,63),(bx,by),
                             (bx+self.side*reach,by-reach*.72),1)
            pygame.draw.line(surface,(47,104,51),(bx,by),
                             (bx-self.side*reach*.55,by+reach*.22),1)


class Stem:
    def draw(self, surface, cx, top, progress, time):
        p=smooth((progress-.22)/.55)
        if p<=0:return
        end=top+440*p
        points=[]
        for i in range(30):
            q=i/29; points.append((cx+math.sin(q*2.2+time*.25)*10*q,top+(end-top)*q))
        pygame.draw.lines(surface,(3,30,17),False,[(x+4,y) for x,y in points],15)
        pygame.draw.lines(surface,(10,72,34),False,points,11)
        pygame.draw.lines(surface,(67,133,64),False,[(x-2,y) for x,y in points],3)


class Rose:
    def __init__(self):
        self.petals=[]
        for layer,total in [(0,5),(1,7),(2,9),(3,11),(4,13),(5,15)]:
            for i in range(total): self.petals.append(Petal(layer,i,total,random.random()))
        self.leaves=[Leaf(-1,180,155),Leaf(1,285,135),Leaf(-1,365,105)]
        self.stem=Stem()

    def draw(self, surface, cx, cy, progress, time):
        self.stem.draw(surface,cx,cy+125,progress,time)
        for leaf in self.leaves: leaf.draw(surface,cx,cy+145,progress,time)
        sepal_progress=smooth((progress-.24)/.38)
        if sepal_progress:
            for i in range(5):
                a=math.tau*i/5-time*.01
                tip=(cx+math.cos(a)*92*sepal_progress,
                     cy+math.sin(a)*52*sepal_progress+35)
                pygame.draw.polygon(surface,(12,74,35),[
                    (cx-8,cy+92),(cx+8,cy+92),tip])
                pygame.draw.line(surface,(57,119,48),(cx,cy+91),tip,2)
        # Rear-to-front order creates a convincing layered bowl.
        for p in sorted(self.petals,key=lambda p:(-p.layer, math.sin(p.angle))):
            p.draw(surface,cx,cy,progress,time)
        # Nested inner petals create the folded, tightly curled heart of the bloom.
        bloom=smooth((progress-.38)/.5)
        if bloom:
            for i in range(7):
                a=time*.13+i*math.tau/7
                radius=(8+i*3)*bloom
                x=cx+math.cos(a)*radius; y=cy+math.sin(a)*radius*.65
                color=(min(255,190+i*8),20+i*3,46+i*4)
                pygame.draw.ellipse(surface,color,
                    pygame.Rect(int(x-8*bloom),int(y-13*bloom),
                                max(1,int(16*bloom)),max(1,int(26*bloom))))


class AnimationController:
    def __init__(self): self.elapsed=0; self.speed=1; self.paused=False; self.effects=True
    @property
    def progress(self): return clamp(self.elapsed/4)
    def update(self,dt):
        if not self.paused:self.elapsed+=dt*self.speed
    def restart(self): self.elapsed=0


def main():
    pygame.init(); pygame.display.set_caption('A Rose Awakens — Live Rose Animation')
    flags=pygame.RESIZABLE
    screen=pygame.display.set_mode((720,1280),flags)
    clock=pygame.time.Clock(); controller=AnimationController(); rose=Rose()
    title_font=pygame.font.SysFont('georgia',30)
    subtitle_font=pygame.font.SysFont('georgia',15,italic=True)
    glow=glow_surface(190,(255,32,58),.75); gold_glow=glow_surface(100,(255,157,65),.6)
    design=pygame.Surface((1080,1920),pygame.SRCALPHA)
    running=True; print('Application started. Press SPACE to restart, P to pause, M for effects, UP/DOWN for speed, ESC to exit.')
    while running:
        dt=clock.tick(60)/1000
        for e in pygame.event.get():
            if e.type==pygame.QUIT: running=False
            elif e.type==pygame.VIDEORESIZE: screen=pygame.display.set_mode(e.size,flags)
            elif e.type==pygame.KEYDOWN:
                if e.key==pygame.K_ESCAPE: running=False
                elif e.key==pygame.K_SPACE: controller.restart()
                elif e.key==pygame.K_p: controller.paused=not controller.paused
                elif e.key==pygame.K_m: controller.effects=not controller.effects
                elif e.key==pygame.K_UP: controller.speed=min(3,controller.speed+.1)
                elif e.key==pygame.K_DOWN: controller.speed=max(.2,controller.speed-.1)
        controller.update(dt); progress=controller.progress; time=controller.elapsed
        w,h=screen.get_size(); scale=min(w/1080,h/1920); cx=w/2; cy=h*.405
        # Letterbox-safe composition, maintaining the 9:16 visual layout.
        screen.fill(BLACK)
        if controller.effects:
            g=pygame.transform.smoothscale(glow,(int(glow.get_width()*scale),int(glow.get_height()*scale)))
            screen.blit(g,(int(cx-g.get_width()/2),int(cy-g.get_height()/2)),special_flags=pygame.BLEND_RGB_ADD)
            gg=pygame.transform.smoothscale(gold_glow,(int(gold_glow.get_width()*scale),int(gold_glow.get_height()*scale)))
            screen.blit(gg,(int(cx-gg.get_width()/2),int(cy-gg.get_height()/2)),special_flags=pygame.BLEND_RGB_ADD)
        # Render rose to a transparent design surface for consistent scaling.
        design.fill((0,0,0,0))
        design_cx=540; design_cy=770
        rose.draw(design,design_cx,design_cy,progress,time)
        # A gentle camera orbit is suggested through a subtle horizontal oscillation.
        scaled=pygame.transform.smoothscale(design,(int(1080*scale),int(1920*scale)))
        breathing=1+.006*math.sin(time*.45)
        if breathing != 1:
            scaled=pygame.transform.smoothscale(
                scaled,(max(1,int(scaled.get_width()*breathing)),
                        max(1,int(scaled.get_height()*breathing))))
        screen.blit(scaled,(int(cx-scaled.get_width()/2+math.sin(time*.25)*2),
                            int(h/2-scaled.get_height()/2)))
        title=title_font.render('A ROSE AWAKENS',True,(228,198,161))
        subtitle=subtitle_font.render('a study in light, motion & bloom',True,(145,121,105))
        title.set_alpha(int(235*smooth(progress/.18)))
        subtitle.set_alpha(int(190*smooth(progress/.3)))
        screen.blit(title,(int(cx-title.get_width()/2),int(h*.065)))
        screen.blit(subtitle,(int(cx-subtitle.get_width()/2),int(h*.065)+title.get_height()+5))
        pygame.display.flip()
    pygame.quit(); print('\nApplication closed.')

if __name__=='__main__': main()
