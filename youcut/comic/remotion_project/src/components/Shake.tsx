// Shake — wrapper que aplica translate(x, y) por janelas de impacto.
//
// Para cada `shake` em props.shakes, aplica um deslocamento sinusoidal
// decaindo linearmente ao longo de SHAKE_DURATION_SEC. Sem efeito quando
// o frame atual está fora de qualquer janela de shake.

import { useCurrentFrame, useVideoConfig } from "remotion";
import type { Shake as ShakeData } from "../types";

const SHAKE_DURATION_SEC = 0.3;

interface ShakeProps {
  shakes: ShakeData[];
  children: React.ReactNode;
}

export const Shake: React.FC<ShakeProps> = ({ shakes, children }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;

  let dx = 0;
  let dy = 0;
  for (const s of shakes) {
    const dt = t - s.at_sec;
    if (dt < 0 || dt > SHAKE_DURATION_SEC) continue;
    const decay = 1 - dt / SHAKE_DURATION_SEC;
    const amp = (s.intensity ?? 1) * 18 * decay;
    dx += Math.sin(dt * 60) * amp;
    dy += Math.cos(dt * 70) * amp * 0.8;
  }

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        transform: `translate(${dx}px, ${dy}px)`,
      }}
    >
      {children}
    </div>
  );
};
