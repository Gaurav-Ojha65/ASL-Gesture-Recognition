import { createContext, useContext, useState, type ReactNode } from 'react';

interface CameraStatusContextValue {
  cameraActive: boolean;
  setCameraActive: (active: boolean) => void;
}

const CameraStatusContext = createContext<CameraStatusContextValue>({
  cameraActive: false,
  setCameraActive: () => {},
});

export function CameraStatusProvider({ children }: { children: ReactNode }) {
  const [cameraActive, setCameraActive] = useState(false);
  return (
    <CameraStatusContext.Provider value={{ cameraActive, setCameraActive }}>
      {children}
    </CameraStatusContext.Provider>
  );
}

export function useCameraStatus() {
  return useContext(CameraStatusContext);
}
