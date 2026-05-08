// ComicVideo — composição raiz do engine `youcut comic --engine remotion`.
//
// Camadas (ordem de pintura):
//   1. <AbsoluteFill background_color>
//   2. <Audio> (trilha original do vídeo de entrada)
//   3. mapeamento de `props.scenes` em <Sequence> consecutivos
//      cada Sequence renderiza um <Scene/> com Ken Burns, transição,
//      shake, lip-sync e idle blink/breathe.
//
// O `render.mjs` reescreve `audio_path` e os paths de cada character
// (anchor / mouth_sheet) para basenames relativos ao publicDir antes de
// chamar renderMedia, então sempre podemos usar `staticFile()`.

import { AbsoluteFill, Audio, Sequence, staticFile, useVideoConfig } from "remotion";
import { Scene } from "./components/Scene";
import type { InputProps } from "./types";

const resolveAudioSrc = (audioPath: string): string =>
  audioPath ? staticFile(audioPath) : "";

export const ComicVideo: React.FC<InputProps> = ({
  audio_path,
  background_color,
  characters,
  scenes,
}) => {
  const { fps } = useVideoConfig();
  const audioSrc = resolveAudioSrc(audio_path);
  const charsMap = characters ?? {};

  return (
    <AbsoluteFill style={{ backgroundColor: background_color ?? "#000000" }}>
      {audioSrc ? <Audio src={audioSrc} /> : null}
      {(scenes ?? []).map((scene, idx) => {
        const fromFrame = Math.max(0, Math.round(scene.start_sec * fps));
        const durationInFrames = Math.max(
          1,
          Math.round((scene.end_sec - scene.start_sec) * fps),
        );
        return (
          <Sequence
            key={`scene-${idx}`}
            from={fromFrame}
            durationInFrames={durationInFrames}
          >
            <Scene scene={scene} characters={charsMap} />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};
