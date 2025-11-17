from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def read_root():
    return {'Hello': 'World'}

# TODO: Complete prediction API
@app.post('/predict_absenteeism/{worker_id}')
def predict_absenteeism(worker_id: int):
    return {'worker_id': worker_id}