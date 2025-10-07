"""
Procesamiento de video para Computer Vision.
"""

from typing import Optional, List, Tuple, Union, Dict, Any, Callable
import numpy as np
from pathlib import Path
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Metadata de un video."""
    path: Path
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    codec: str


class VideoProcessor:
    """Procesador de videos.
    
    Parameters
    ----------
    input_path : Union[str, Path]
        Ruta del video de entrada.
    output_path : Optional[Union[str, Path]]
        Ruta del video de salida.
    fps : Optional[int]
        FPS objetivo para procesamiento.
    resize : Optional[Tuple[int, int]]
        Tamaño para redimensionar (width, height).
    """
    
    def __init__(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        fps: Optional[int] = None,
        resize: Optional[Tuple[int, int]] = None
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path) if output_path else None
        self.fps = fps
        self.resize = resize
        self.metadata = None
        self._load_metadata()
    
    def _load_metadata(self):
        """Carga metadata del video."""
        try:
            import cv2
        except ImportError:
            logger.error("opencv-python required for video processing")
            return
        
        cap = cv2.VideoCapture(str(self.input_path))
        
        self.metadata = VideoMetadata(
            path=self.input_path,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=cap.get(cv2.CAP_PROP_FPS),
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            duration=cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            codec=self._get_codec(cap)
        )
        
        cap.release()
        logger.info(f"Video loaded: {self.metadata.total_frames} frames at {self.metadata.fps:.2f} FPS")
    
    def _get_codec(self, cap) -> str:
        """Obtiene codec del video."""
        try:
            import cv2
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            return codec
        except:
            return "unknown"
    
    def process(
        self,
        frame_processor: Callable[[np.ndarray, int], np.ndarray],
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        show_progress: bool = True
    ) -> bool:
        """Procesa video aplicando función a cada frame.
        
        Parameters
        ----------
        frame_processor : Callable[[np.ndarray, int], np.ndarray]
            Función que procesa cada frame.
        start_frame : int
            Frame inicial.
        end_frame : Optional[int]
            Frame final.
        show_progress : bool
            Si mostrar barra de progreso.
            
        Returns
        -------
        bool
            True si el procesamiento fue exitoso.
        """
        try:
            import cv2
            from tqdm import tqdm
        except ImportError:
            logger.error("opencv-python and tqdm required")
            return False
        
        # Abrir video de entrada
        cap = cv2.VideoCapture(str(self.input_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Configurar video de salida si se especifica
        writer = None
        if self.output_path:
            fps = self.fps or self.metadata.fps
            
            if self.resize:
                width, height = self.resize
            else:
                width, height = self.metadata.width, self.metadata.height
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                fps,
                (width, height)
            )
        
        # Determinar rango de frames
        end_frame = end_frame or self.metadata.total_frames
        total_frames = end_frame - start_frame
        
        # Procesar frames
        iterator = tqdm(range(total_frames)) if show_progress else range(total_frames)
        
        for frame_idx in iterator:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Redimensionar si es necesario
            if self.resize:
                frame = cv2.resize(frame, self.resize)
            
            # Aplicar procesamiento
            processed_frame = frame_processor(frame, start_frame + frame_idx)
            
            # Escribir frame procesado
            if writer:
                writer.write(processed_frame)
        
        # Limpiar
        cap.release()
        if writer:
            writer.release()
            logger.info(f"Processed video saved to {self.output_path}")
        
        return True
    
    def extract_frames(
        self,
        output_dir: Union[str, Path],
        interval: int = 1,
        format: str = 'jpg'
    ) -> List[Path]:
        """Extrae frames del video.
        
        Parameters
        ----------
        output_dir : Union[str, Path]
            Directorio de salida.
        interval : int
            Intervalo entre frames a extraer.
        format : str
            Formato de imagen.
            
        Returns
        -------
        List[Path]
            Lista de rutas de frames extraídos.
        """
        try:
            import cv2
        except ImportError:
            logger.error("opencv-python required")
            return []
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(self.input_path))
        frame_paths = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % interval == 0:
                # Guardar frame
                frame_path = output_dir / f"frame_{frame_count:06d}.{format}"
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(frame_path)
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {len(frame_paths)} frames to {output_dir}")
        
        return frame_paths
    
    def create_video_from_frames(
        self,
        frame_paths: List[Path],
        output_path: Union[str, Path],
        fps: int = 30
    ) -> bool:
        """Crea video desde frames.
        
        Parameters
        ----------
        frame_paths : List[Path]
            Lista de rutas de frames.
        output_path : Union[str, Path]
            Ruta del video de salida.
        fps : int
            FPS del video.
            
        Returns
        -------
        bool
            True si fue exitoso.
        """
        try:
            import cv2
        except ImportError:
            logger.error("opencv-python required")
            return False
        
        if not frame_paths:
            logger.error("No frames provided")
            return False
        
        # Leer primer frame para obtener dimensiones
        first_frame = cv2.imread(str(frame_paths[0]))
        height, width = first_frame.shape[:2]
        
        # Crear writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )
        
        # Escribir frames
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                writer.write(frame)
        
        writer.release()
        logger.info(f"Video created at {output_path}")
        
        return True


class FrameExtractor:
    """Extractor de frames específicos.
    
    Parameters
    ----------
    video_path : Union[str, Path]
        Ruta del video.
    """
    
    def __init__(self, video_path: Union[str, Path]):
        self.video_path = Path(video_path)
        self.cap = None
        self._open()
    
    def _open(self):
        """Abre el video."""
        try:
            import cv2
            self.cap = cv2.VideoCapture(str(self.video_path))
        except ImportError:
            logger.error("opencv-python required")
    
    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Obtiene un frame específico.
        
        Parameters
        ----------
        frame_number : int
            Número del frame.
            
        Returns
        -------
        Optional[np.ndarray]
            Frame o None si no existe.
        """
        if self.cap is None:
            return None
        
        self.cap.set(1, frame_number)  # CAP_PROP_POS_FRAMES
        ret, frame = self.cap.read()
        
        return frame if ret else None
    
    def get_frames(
        self,
        frame_numbers: List[int]
    ) -> List[Optional[np.ndarray]]:
        """Obtiene múltiples frames.
        
        Parameters
        ----------
        frame_numbers : List[int]
            Números de frames.
            
        Returns
        -------
        List[Optional[np.ndarray]]
            Lista de frames.
        """
        return [self.get_frame(n) for n in frame_numbers]
    
    def get_frame_at_time(self, time_seconds: float) -> Optional[np.ndarray]:
        """Obtiene frame en un tiempo específico.
        
        Parameters
        ----------
        time_seconds : float
            Tiempo en segundos.
            
        Returns
        -------
        Optional[np.ndarray]
            Frame o None.
        """
        if self.cap is None:
            return None
        
        fps = self.cap.get(5)  # CAP_PROP_FPS
        frame_number = int(time_seconds * fps)
        return self.get_frame(frame_number)
    
    def close(self):
        """Cierra el video."""
        if self.cap:
            self.cap.release()
    
    def __del__(self):
        """Destructor."""
        self.close()


class VideoWriter:
    """Escritor de video.
    
    Parameters
    ----------
    output_path : Union[str, Path]
        Ruta de salida.
    fps : int
        Frames por segundo.
    frame_size : Tuple[int, int]
        Tamaño del frame (width, height).
    codec : str
        Codec de video.
    """
    
    def __init__(
        self,
        output_path: Union[str, Path],
        fps: int = 30,
        frame_size: Optional[Tuple[int, int]] = None,
        codec: str = 'mp4v'
    ):
        self.output_path = Path(output_path)
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self.writer = None
    
    def _init_writer(self, frame: np.ndarray):
        """Inicializa el writer con el primer frame."""
        try:
            import cv2
        except ImportError:
            logger.error("opencv-python required")
            return
        
        if self.frame_size is None:
            height, width = frame.shape[:2]
            self.frame_size = (width, height)
        
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            self.frame_size
        )
    
    def write_frame(self, frame: np.ndarray):
        """Escribe un frame.
        
        Parameters
        ----------
        frame : np.ndarray
            Frame a escribir.
        """
        if self.writer is None:
            self._init_writer(frame)
        
        if self.writer:
            # Redimensionar si es necesario
            if self.frame_size and frame.shape[:2][::-1] != self.frame_size:
                try:
                    import cv2
                    frame = cv2.resize(frame, self.frame_size)
                except ImportError:
                    pass
            
            self.writer.write(frame)
    
    def write_frames(self, frames: List[np.ndarray]):
        """Escribe múltiples frames.
        
        Parameters
        ----------
        frames : List[np.ndarray]
            Lista de frames.
        """
        for frame in frames:
            self.write_frame(frame)
    
    def close(self):
        """Cierra el writer."""
        if self.writer:
            self.writer.release()
            logger.info(f"Video saved to {self.output_path}")
    
    def __enter__(self):
        """Context manager entrada."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager salida."""
        self.close()


def apply_model_to_video(
    video_path: Union[str, Path],
    model: Any,
    output_path: Union[str, Path],
    task_type: str = 'detection',
    visualize: bool = True,
    class_names: Optional[List[str]] = None,
    confidence_threshold: float = 0.5
) -> Dict[str, Any]:
    """Aplica modelo a video completo.
    
    Parameters
    ----------
    video_path : Union[str, Path]
        Ruta del video.
    model : Any
        Modelo a aplicar.
    output_path : Union[str, Path]
        Ruta de salida.
    task_type : str
        Tipo de tarea ('detection', 'segmentation', 'classification').
    visualize : bool
        Si visualizar resultados en video.
    class_names : Optional[List[str]]
        Nombres de clases.
    confidence_threshold : float
        Umbral de confianza.
        
    Returns
    -------
    Dict[str, Any]
        Resultados del procesamiento.
    """
    from .utils import draw_bounding_boxes, draw_segmentation_masks
    
    processor = VideoProcessor(video_path, output_path if visualize else None)
    results = {
        'video_path': str(video_path),
        'output_path': str(output_path) if visualize else None,
        'frame_results': []
    }
    
    def process_frame(frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Procesa un frame."""
        # Aplicar modelo
        if task_type == 'detection':
            predictions = model.predict(frame)
            
            # Filtrar por confianza
            if 'scores' in predictions:
                keep = predictions['scores'] > confidence_threshold
                predictions = {
                    'boxes': predictions['boxes'][keep],
                    'labels': predictions['labels'][keep],
                    'scores': predictions['scores'][keep]
                }
            
            # Guardar resultados
            results['frame_results'].append({
                'frame': frame_idx,
                'detections': len(predictions['boxes']),
                'predictions': predictions
            })
            
            # Visualizar si se solicita
            if visualize:
                frame = draw_bounding_boxes(
                    frame,
                    predictions['boxes'],
                    predictions['labels'],
                    predictions['scores'],
                    class_names
                )
        
        elif task_type == 'segmentation':
            mask = model.predict(frame)
            
            results['frame_results'].append({
                'frame': frame_idx,
                'mask_shape': mask.shape
            })
            
            if visualize:
                frame = draw_segmentation_masks(
                    frame,
                    mask,
                    class_names
                )
        
        elif task_type == 'classification':
            prediction = model.predict(frame)
            
            results['frame_results'].append({
                'frame': frame_idx,
                'prediction': prediction
            })
            
            if visualize:
                # Añadir texto con predicción
                try:
                    import cv2
                    text = class_names[prediction] if class_names else f"Class {prediction}"
                    cv2.putText(
                        frame,
                        text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                except ImportError:
                    pass
        
        return frame
    
    # Procesar video
    success = processor.process(process_frame)
    
    if success:
        results['status'] = 'completed'
        results['total_frames'] = len(results['frame_results'])
    else:
        results['status'] = 'failed'
    
    return results


def track_objects(
    video_path: Union[str, Path],
    detector: Any,
    output_path: Optional[Union[str, Path]] = None,
    tracker_type: str = 'sort',
    max_age: int = 30,
    min_hits: int = 3,
    iou_threshold: float = 0.3
) -> Dict[str, Any]:
    """Rastrea objetos en video.
    
    Parameters
    ----------
    video_path : Union[str, Path]
        Ruta del video.
    detector : Any
        Detector de objetos.
    output_path : Optional[Union[str, Path]]
        Ruta de salida.
    tracker_type : str
        Tipo de tracker ('sort', 'deep_sort').
    max_age : int
        Máximo de frames sin detección.
    min_hits : int
        Mínimo de detecciones para confirmar track.
    iou_threshold : float
        Umbral IoU para asociación.
        
    Returns
    -------
    Dict[str, Any]
        Resultados del tracking.
    """
    processor = VideoProcessor(video_path, output_path)
    
    # Tracker simple basado en IoU
    tracks = {}
    next_track_id = 0
    results = {
        'video_path': str(video_path),
        'tracks': [],
        'frame_tracks': []
    }
    
    def process_frame(frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Procesa frame con tracking."""
        nonlocal next_track_id
        
        # Detectar objetos
        detections = detector.predict(frame)
        
        # Tracking simple (placeholder - implementación completa requeriría SORT/DeepSORT)
        frame_tracks = []
        for i, box in enumerate(detections.get('boxes', [])):
            # Asignar ID de track (simplificado)
            track_id = next_track_id
            next_track_id += 1
            
            frame_tracks.append({
                'track_id': track_id,
                'box': box.tolist(),
                'label': detections['labels'][i] if 'labels' in detections else 0,
                'score': detections['scores'][i] if 'scores' in detections else 1.0
            })
            
            # Dibujar track
            if output_path:
                try:
                    import cv2
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"ID: {track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
                except ImportError:
                    pass
        
        results['frame_tracks'].append({
            'frame': frame_idx,
            'tracks': frame_tracks
        })
        
        return frame
    
    # Procesar video
    success = processor.process(process_frame)
    
    if success:
        results['status'] = 'completed'
        results['total_tracks'] = next_track_id
    else:
        results['status'] = 'failed'
    
    return results