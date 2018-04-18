from channels.generic.websocket import WebsocketConsumer
from asgiref.sync import async_to_sync
import json
from .network_tools import createNetwork, train


class ChatConsumer(WebsocketConsumer):
    def connect(self):
    	#self.room_name = self.scope['url_route']['kwargs']['room_name']
    	#self.room_group_name = 'room_%s' % self.room_name
        print("connected to socket")
        self.accept()

    def disconnect(self,data):
        print('disconnect: %s' % data)

    def receive(self, text_data):
        data = json.loads(text_data)
        if (data['action']=='train'):
        	self.network_train(data)

    # def train(model, model_name, epochs=5, input_type="2d", batch_size=128):
    def network_train(self,data):
      returnDict=dict()
      print('train data: ',data)
      print('model create')
      try:
        (model,input_type) = createNetwork(data["model"])
      except Exception:
        returnDict["action"] = 'error'
        returnDict["message"] = "Ошибка создания модели"
        print("Ошибка создания модели")
        self.send(json.dumps(returnDict))
        self.disconnect("Ошибка создания модели")
        return

      epochs = int(data['epochs'])
      batch_size = int(data['batch-size'])
      interval = [int(data['from']),int(data['till'])]
      
      #try:
      (av_err,pc) = train(model,data["model"],epochs=epochs,interval=interval,input_type=input_type, batch_size=batch_size,socket=self)
      #except Exception:
      #  returnDict["action"] = "error"
      #  returnDict["message"] = "Ошибка во время обучения"
      #  print("Ошибка во время обучения")
      #  self.send(json.dumps(returnDict))
      #  self.disconnect("Ошибка во время обучения")
      #else:
      returnDict['av-err']=av_err
      returnDict['percent']=pc
      returnDict['action']='test-end'
      
      self.send(json.dumps(returnDict))
      self.disconnect('training finished')
