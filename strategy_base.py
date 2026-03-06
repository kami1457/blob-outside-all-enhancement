
import cv2
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('image_processing')

class BaseStrategy:
    _registry = {}

    def __init__(self, **kwargs):
        self.params = kwargs
        self._cache = {}
        self._cache_enabled = self.params.get('cache_enabled', False)
        self._cache_size = self.params.get('cache_size', 100)
        self._logger = logging.getLogger(f'image_processing.{self.__class__.__name__}')
        self._logger.info(f'初始化策略: {self.__class__.__name__}，参数: {kwargs}')

    @classmethod
    def register_strategy(cls, name):
        def decorator(strategy_class):
            cls._registry[name] = strategy_class
            return strategy_class
        return decorator

    @classmethod
    def get_strategy(cls, name):
        if name not in cls._registry:
            raise ValueError(f"未知策略: {name}")
        return cls._registry[name]

    def process(self, frame):
        raise NotImplementedError("子类必须实现 process 方法")

    def _compose_images(self, frame, items):
        composite_img = np.zeros(frame.shape, dtype=np.uint8)
        filled_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for item in items:
            result_img = item['result']
            gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
            _, current_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            fill_mask = cv2.bitwise_and(current_mask, cv2.bitwise_not(filled_mask))
            composite_img[fill_mask == 255] = result_img[fill_mask == 255]
            filled_mask = cv2.bitwise_or(filled_mask, fill_mask)

        return composite_img

    def _validate_params(self, required_params=None):
        if required_params:
            for param in required_params:
                if param not in self.params:
                    raise ValueError(f"缺少必要参数: {param}")

    def _validate_frame(self, frame):
        if frame is None:
            raise ValueError("输入图像为None")
        if len(frame.shape) != 3:
            raise ValueError("输入图像必须是BGR格式")
        if frame.shape[2] != 3:
            raise ValueError("输入图像必须是3通道BGR格式")

    def _compute_frame_hash(self, frame):
        resized = cv2.resize(frame, (32, 32))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        mean = gray.mean()
        hash_value = 0
        for i in range(32):
            for j in range(32):
                hash_value = hash_value * 2 + (1 if gray[i, j] > mean else 0)
        return hash_value

    def _get_from_cache(self, frame):
        if not self._cache_enabled:
            return None
        frame_hash = self._compute_frame_hash(frame)
        return self._cache.get(frame_hash)

    def _add_to_cache(self, frame, result):
        if not self._cache_enabled:
            return
        frame_hash = self._compute_frame_hash(frame)
        if len(self._cache) >= self._cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[frame_hash] = result

@BaseStrategy.register_strategy('composite')
class CompositeStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.strategies = []
        strategy_configs = self.params.get('strategies', [])
        for config in strategy_configs:
            strategy_name = config['name']
            strategy_params = config.get('params', {})
            strategy_class = BaseStrategy.get_strategy(strategy_name)
            self.strategies.append(strategy_class(**strategy_params))

    def process(self, frame):
        self._validate_frame(frame)
        current_frame = frame.copy()
        all_info = []
        
        for strategy in self.strategies:
            current_frame, info = strategy.process(current_frame)
            all_info.extend(info)
        
        return current_frame, all_info