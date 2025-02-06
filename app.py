import os
import pandas as pd
from typing import List
from catboost import CatBoostClassifier
from fastapi import FastAPI
from schema import PostGet
from datetime import datetime
from sqlalchemy import create_engine
from loguru import logger


app = FastAPI()
DATABASE_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"

def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # Проверяем, где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/Dev_notebooks'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_features():
    """Функция для загрузки данных и фичей."""
    logger.info('loading liked posts')
    liked_post_query = """
        SELECT DISTINCT post_id, user_id
        FROM public.feed_data
        WHERE action = 'like'
    """
    liked_posts = batch_load_sql(liked_post_query)

    logger.info('loading post features')
    posts_features = pd.read_sql(
        "SELECT * FROM public.nm_lesson_22_post", con=DATABASE_URL
    )

    logger.info('loading user features')
    user_features = pd.read_sql(
        "SELECT * FROM public.user_data", con=DATABASE_URL
    )
    return [liked_posts, posts_features, user_features]

def load_models():
    """Функция для загрузки модели CatBoost."""
    logger.info("loading Dev_notebooks")
    model_path = get_model_path("Model/catboost_model.cbm")
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model

# Загружаем модель и фичи при старте сервиса
model = load_models()
features = load_features()
logger.info("service is up and running")

def get_recommended_feed(id: int, time: datetime, limit: int):
    """
    Функция для формирования рекомендательной ленты для пользователя.

    Параметры:
    - id: ID пользователя.
    - time: Время, на которое строится рекомендация.
    - limit: Количество постов в рекомендации.
    """
    # Загрузка фичей для пользователя
    logger.info(f'user_id: {id}')
    user_features = features[2].loc[features[2].user_id == id]
    if user_features.empty:
        logger.warning(f"No features found for user_id: {id}")
        return []

    user_features = user_features.drop('user_id', axis=1)

    # Загрузка фичей для постов
    logger.info('processing post features')
    posts_features = features[1].drop(['index', 'text'], axis=1, errors='ignore')
    content = features[1][['post_id', 'text', 'topic']]

    # Объединение фичей пользователя и постов
    logger.info('merging user and post features')
    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    user_posts_features = posts_features.assign(**add_user_features)
    user_posts_features = user_posts_features.set_index('post_id')

    # Добавление временных фичей (час и месяц)
    logger.info("adding time features")
    user_posts_features['hour'] = time.hour
    user_posts_features['month'] = time.month

    # Формирование предсказаний модели
    logger.info("predicting probabilities")
    user_posts_features['predicts'] = model.predict_proba(user_posts_features)[:, 1]

    # Убираем посты, которые пользователь уже лайкнул
    logger.info('filtering liked posts')
    liked_posts = features[0]
    liked_post_ids = liked_posts[liked_posts.user_id == id].post_id.values
    filtered = user_posts_features[~user_posts_features.index.isin(liked_post_ids)]

    # Сортировка по вероятности и выборка топ-N постов
    logger.info('sorting and selecting top posts')
    recommended_posts = filtered.sort_values('predicts', ascending=False).head(limit).index

    # Формируем итоговый список рекомендаций
    return [
        PostGet(
            id=i,
            text=content.loc[content.post_id == i, 'text'].values[0],
            topic=content.loc[content.post_id == i, 'topic'].values[0]
        ) for i in recommended_posts
    ]

@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_post(id: int, time: datetime, limit: int = 10) -> List[PostGet]:
    """Эндпоинт для получения рекомендаций."""
    return get_recommended_feed(id, time, limit)