import GameOfLife3D from "./GameOfLife3D";

export default function Home() {
  return (
    <div className="w-full h-full relative">
      <div className="pointer-events-none absolute top-0 left-0 w-full h-full bg-black/20 z-10 backdrop-blur-md"></div>
      <GameOfLife3D />
    </div>
  );
}
