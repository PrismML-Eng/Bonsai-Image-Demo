<script lang="ts">
  import { onMount } from 'svelte';

  export type GraphicMode = 'loading' | 'speaking' | 'thinking' | 'waiting';

  export type AudioLevelSource = { current: number };

  let {
    mode = 'waiting',
    audioLevelSource,
  }: { mode?: GraphicMode; audioLevelSource?: AudioLevelSource } = $props();

  let canvas: HTMLCanvasElement | undefined = $state();
  const modeRef: { current: GraphicMode } = { current: 'waiting' };
  const audioLevelHolder: { source: AudioLevelSource } = { source: { current: 0 } };

  $effect(() => {
    modeRef.current = mode;
    audioLevelHolder.source = audioLevelSource ?? audioLevelHolder.source;
  });

  const SPEEDS: Record<GraphicMode, number> = {
    loading: 0.7,
    waiting: 0.28,
    thinking: 1.8,
    speaking: 6.5,
  };

  const ENERGY: Record<GraphicMode, number> = {
    loading: 0.45,
    waiting: 0.35,
    thinking: 0.7,
    speaking: 1,
  };

  onMount(() => {
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const size = 340;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    type TrailPoint = { x: number; y: number; life: number };
    type Particle = {
      offset: number;
      orbit: number;
      tilt: number;
      size: number;
      hueShift: number;
      trail: TrailPoint[];
    };

    const particles: Particle[] = Array.from({ length: 28 }, (_, i) => ({
      offset: (i / 28) * Math.PI * 2,
      orbit: 52 + (i % 5) * 18,
      tilt: 0.55 + (i % 3) * 0.22,
      size: 1.8 + (i % 4) * 0.9,
      hueShift: (i % 7) * 0.08,
      trail: [],
    }));

    const rings = [
      { radius: 78, tilt: 0.42, width: 1.2, direction: 1 },
      { radius: 98, tilt: 0.68, width: 1, direction: -1 },
      { radius: 118, tilt: 0.35, width: 0.8, direction: 1 },
    ];

    let angle = 0;
    let speed = SPEEDS.waiting;
    let energy = ENERGY.waiting;
    let volumeScale = 1;
    let breathe = 0;
    let shimmer = 0;
    let last = performance.now();
    let frameId = 0;

    const cx = size / 2;
    const cy = size / 2;

    const rgba = (r: number, g: number, b: number, a: number) =>
      `rgba(${r | 0}, ${g | 0}, ${b | 0}, ${a})`;

    const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

    const drawEllipseRing = (
      radius: number,
      tilt: number,
      rotation: number,
      alpha: number,
      lineWidth: number,
    ) => {
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(rotation);
      ctx.scale(1, tilt);

      const grad = ctx.createLinearGradient(-radius, 0, radius, 0);
      grad.addColorStop(0, rgba(16, 163, 127, 0));
      grad.addColorStop(0.35, rgba(56, 220, 180, alpha * 0.7));
      grad.addColorStop(0.5, rgba(180, 255, 230, alpha));
      grad.addColorStop(0.65, rgba(56, 220, 180, alpha * 0.7));
      grad.addColorStop(1, rgba(16, 163, 127, 0));

      ctx.beginPath();
      ctx.arc(0, 0, radius, 0, Math.PI * 2);
      ctx.strokeStyle = grad;
      ctx.lineWidth = lineWidth;
      ctx.shadowColor = rgba(56, 220, 180, alpha * 0.8);
      ctx.shadowBlur = 8 + energy * 14;
      ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.restore();
    };

    const drawTendrils = (time: number, intensity: number) => {
      const count = 8;
      for (let i = 0; i < count; i += 1) {
        const base = (i / count) * Math.PI * 2 + time * 0.15;
        const length = 38 + intensity * 42 + Math.sin(time * 1.4 + i) * 10;
        const wobble = Math.sin(time * 2.2 + i * 1.7) * (6 + intensity * 10);

        ctx.beginPath();
        ctx.moveTo(cx, cy);
        const steps = 12;
        for (let s = 1; s <= steps; s += 1) {
          const t = s / steps;
          const r = length * t;
          const a = base + wobble * t * 0.08;
          ctx.lineTo(cx + Math.cos(a) * r, cy + Math.sin(a) * r);
        }

        const alpha = 0.04 + intensity * 0.1;
        ctx.strokeStyle = rgba(120, 255, 210, alpha * (1 - i * 0.04));
        ctx.lineWidth = 1.2 + intensity * 1.5;
        ctx.lineCap = 'round';
        ctx.stroke();
      }
    };

    const tick = (now: number) => {
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;

      const mode = modeRef.current;
      const audioLevel = mode === 'speaking' ? audioLevelHolder.source.current : 0;

      const targetSpeed = SPEEDS[mode];
      const targetEnergy = ENERGY[mode];
      const targetScale = mode === 'speaking' ? 0.58 + audioLevel * 0.72 : 1;

      speed += (targetSpeed - speed) * Math.min(1, dt * 3.2);
      energy += (targetEnergy - energy) * Math.min(1, dt * 2.8);
      volumeScale += (targetScale - volumeScale) * Math.min(1, dt * 14);
      angle += speed * dt;
      breathe += dt * (0.65 + speed * 0.14);
      shimmer += dt * (1.2 + speed * 0.25);

      ctx.clearRect(0, 0, size, size);

      ctx.save();
      ctx.translate(cx, cy);
      ctx.scale(volumeScale, volumeScale);
      ctx.translate(-cx, -cy);

      // Ambient bloom
      const pulse = 0.55 + Math.sin(breathe) * 0.12;
      const bloom = ctx.createRadialGradient(cx, cy, 4, cx, cy, 130 * pulse);
      bloom.addColorStop(0, rgba(80, 255, 200, 0.1 + energy * 0.14));
      bloom.addColorStop(0.35, rgba(16, 163, 127, 0.08 + energy * 0.1));
      bloom.addColorStop(1, rgba(16, 163, 127, 0));
      ctx.fillStyle = bloom;
      ctx.beginPath();
      ctx.arc(cx, cy, 140, 0, Math.PI * 2);
      ctx.fill();

      drawTendrils(shimmer, energy);

      for (const [index, ring] of rings.entries()) {
        const wobble =
          Math.sin(breathe * 1.1 + index) * (2 + speed * 0.35) +
          Math.sin(shimmer * 0.8 + index * 2) * energy * 3;
        drawEllipseRing(
          ring.radius + wobble,
          ring.tilt + Math.sin(shimmer + index) * 0.04 * energy,
          angle * ring.direction * (0.35 + index * 0.15),
          0.12 + energy * 0.22,
          ring.width + energy * 0.6,
        );
      }

      // Particle orbits with comet trails
      for (const particle of particles) {
        const wobble =
          Math.sin(breathe * 1.6 + particle.offset) * (5 + speed * 0.4) +
          Math.cos(shimmer * 1.3 + particle.offset * 2) * energy * 4;
        const orbit = particle.orbit + wobble;
        const a =
          angle * (1.1 + particle.hueShift) +
          particle.offset +
          Math.sin(breathe + particle.offset) * 0.08 * speed;
        const x = cx + Math.cos(a) * orbit;
        const y = cy + Math.sin(a) * orbit * particle.tilt;

        particle.trail.unshift({ x, y, life: 1 });
        if (particle.trail.length > 14) particle.trail.pop();

        for (const [ti, point] of particle.trail.entries()) {
          const life = point.life * (1 - ti / particle.trail.length);
          const trailSize = particle.size * (0.4 + life * 0.9) + speed * 0.08;
          const g = 56 + energy * 120 + particle.hueShift * 40;
          ctx.fillStyle = rgba(g, 220, 180, life * (0.08 + energy * 0.2));
          ctx.beginPath();
          ctx.arc(point.x, point.y, trailSize, 0, Math.PI * 2);
          ctx.fill();
        }

        const coreG = lerp(180, 255, energy);
        ctx.fillStyle = rgba(coreG, 255, 230, 0.55 + energy * 0.4);
        ctx.shadowColor = rgba(120, 255, 210, 0.9);
        ctx.shadowBlur = 6 + energy * 16;
        ctx.beginPath();
        ctx.arc(x, y, particle.size + speed * 0.14 + energy * 1.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      }

      // Core nucleus — layered glow
      const corePulse = 10 + Math.sin(breathe * 1.8) * 2 + speed * 0.9 + energy * 4;

      ctx.globalCompositeOperation = 'lighter';
      for (let layer = 0; layer < 3; layer += 1) {
        const r = corePulse * (1.4 - layer * 0.28);
        const alpha = 0.18 - layer * 0.04;
        const coreGlow = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
        coreGlow.addColorStop(0, rgba(220, 255, 245, alpha + energy * 0.25));
        coreGlow.addColorStop(0.5, rgba(56, 220, 180, alpha * 0.6));
        coreGlow.addColorStop(1, rgba(16, 163, 127, 0));
        ctx.fillStyle = coreGlow;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalCompositeOperation = 'source-over';

      ctx.fillStyle = rgba(255, 255, 255, 0.85 + energy * 0.15);
      ctx.beginPath();
      ctx.arc(cx, cy, 5 + energy * 2.5, 0, Math.PI * 2);
      ctx.fill();

      // Speaking shockwave ripples
      if (energy > 0.75) {
        const rippleCount = 3;
        for (let r = 0; r < rippleCount; r += 1) {
          const phase = (shimmer * 1.6 + r * 0.9) % 1;
          const radius = 24 + phase * 110;
          const alpha = (1 - phase) * 0.22 * energy;
          ctx.strokeStyle = rgba(140, 255, 220, alpha);
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(cx, cy, radius, 0, Math.PI * 2);
          ctx.stroke();
        }
      }

      ctx.restore();

      frameId = requestAnimationFrame(tick);
    };

    frameId = requestAnimationFrame(tick);

    return () => cancelAnimationFrame(frameId);
  });
</script>

<canvas bind:this={canvas} class="voice-graphic" aria-hidden="true"></canvas>

<style>
  .voice-graphic {
    display: block;
    margin: 0 auto;
    background: transparent;
    filter: drop-shadow(0 0 28px rgba(16, 163, 127, 0.18));
  }
</style>
