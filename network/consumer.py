from channels.generic.websocket import WebsocketConsumer
import json
from .network_tools import createNetwork, train


class ChatConsumer(WebsocketConsumer):
    def connect(self):
        print("connected to socket")
        self.accept()

    def disconnect(self, data):
        print('disconnect: %s' % data)

    def receive(self, text_data):
        data = json.loads(text_data)
        if (data['action'] == 'train'):
            self.network_train(data)

    def network_train(self, data):
        return_dict = dict()
        print('train data: ', data)
        print('model create')
        try:
            (model, input_type) = create_network(data["model"])
        except Exception:
            return_dict["action"] = 'error'
            return_dict["message"] = "Model create error"
            print("Model create error")
            self.send(json.dumps(returnDict))
            self.disconnect("Model create error")
            return

        epochs = int(data['epochs'])
        batch_size = int(data['batch-size'])
        interval = [int(data['from']), int(data['till'])]

        try:
            (av_err, pc) = train(model, data["model"], epochs=epochs, interval=interval,
                                 input_type=input_type, batch_size=batch_size, socket=self)
        except Exception:
            return_dict["action"] = "error"
            return_dict["message"] = "Error during training"
            print("Error during training")
            self.send(json.dumps(returnDict))
            self.disconnect("Error during training")
        else:
            return_dict['av-err'] = av_err
            return_dict['percent'] = pc
            return_dict['action'] = 'test-end'

            self.send(json.dumps(returnDict))
            self.disconnect('training finished')
