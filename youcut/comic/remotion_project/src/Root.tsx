import { Composition } from "remotion";
import { ComicVideo } from "./ComicVideo";
import type { InputProps } from "./types";

const DEFAULT_FPS = 30;
const DEFAULT_WIDTH = 1080;
const DEFAULT_HEIGHT = 1920;

const DEFAULT_PROPS: InputProps = {
  audio_path: "",
  duration_sec: 1,
  fps: DEFAULT_FPS,
  width: DEFAULT_WIDTH,
  height: DEFAULT_HEIGHT,
  characters: {},
  scenes: [],
  background_color: "#000000",
};

// Cast InputProps -> Record<string, unknown> para satisfazer o `<Composition>`
// genérico do Remotion 4.x. A integridade do shape é garantida pelo modelo
// Pydantic do orquestrador Python (`RemotionInputProps`) e pela interface
// `InputProps` deste módulo.
const ComicVideoComponent =
  ComicVideo as unknown as React.FC<Record<string, unknown>>;

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="ComicVideo"
      component={ComicVideoComponent}
      defaultProps={DEFAULT_PROPS as unknown as Record<string, unknown>}
      durationInFrames={DEFAULT_FPS}
      fps={DEFAULT_FPS}
      width={DEFAULT_WIDTH}
      height={DEFAULT_HEIGHT}
      calculateMetadata={({ props }) => {
        const inputs = props as unknown as InputProps;
        const fps = inputs.fps ?? DEFAULT_FPS;
        const width = inputs.width ?? DEFAULT_WIDTH;
        const height = inputs.height ?? DEFAULT_HEIGHT;
        const durationInFrames = Math.max(
          1,
          Math.round(inputs.duration_sec * fps),
        );
        return { fps, width, height, durationInFrames };
      }}
    />
  );
};
