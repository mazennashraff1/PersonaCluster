from typing import List

from ultralytics import YOLO

from app.models.detector import BoundingBox, PersonDetection

from app import configuration as config


class PersonDetector:

    def __init__(
        self,
        modelName: str,
        confidence_threshold: float,
        device: str | None,
    ):

        self.model = YOLO(modelName)

        self.confidence_threshold = confidence_threshold

        self.device = device

    def detect(self, image) -> List[PersonDetection]:

        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            classes=[config.PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )

        detections = []

        if not results:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        for box in result.boxes:

            coordinates = box.xyxy[0].cpu().numpy()

            confidence = float(box.conf[0].cpu().item())

            x1, y1, x2, y2 = coordinates.astype(int)

            detections.append(
                PersonDetection(
                    bbox=BoundingBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    ),
                    confidence=confidence,
                )
            )

        return detections
