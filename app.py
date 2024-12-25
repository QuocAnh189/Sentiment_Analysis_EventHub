from concurrent import futures
import grpc
from flask import Flask, render_template, request, jsonify
import pickle 
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, GRU, LSTM
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer

import proto.gen.review_pb2 as review_pb2
import proto.gen.review_pb2_grpc as review_pb2_grpc

model = load_model("model/Model.h5")
with open('tokenize.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

class ReviewService(review_pb2_grpc.ReviewServicer):
    def sentiment_analysis(self, request, context):
        try:
            text = request.content
            text_list = [text]
            text_token = tokenizer.texts_to_sequences(text_list)
            text_pad = pad_sequences(text_token, maxlen = 241, padding = 'pre')
            pred = model.predict(text_pad)
            if pred[0][0] > 0.5:
                result = "Positive"
                per = round((pred[0][0])*100 , 2)
            else:
                result = 'Negative'
                per = round((pred[0][0])*100,2) 

            return review_pb2.PredictResponse(
                result=result,
            )
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return review_pb2.PredictResponse()
        

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    review_pb2_grpc.add_ReviewServicer_to_server(ReviewService(), server)
    server.add_insecure_port('[::]:3000')
    print("Server is starting on port 3000...")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
    # app.run(debug=True)