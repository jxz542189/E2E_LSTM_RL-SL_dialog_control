from entity_extractor import EntityExtractor
from encoder import Encoder
from config import Config
from action_manipulator import ActionManipulator
from data_util import DataUtil
from model import Model


class InteractiveSession():
    def __init__(self):

        self.entity_extractor = EntityExtractor()
        self.encoder = Encoder()
        self.action_manipulator = ActionManipulator()
        self.config = Config()
        self.du = DataUtil()
        obs_size = self.config.u_embed_size + self.config.vocab_size + self.config.feature_size
        self.action_templates = self.action_manipulator.get_action_templates()
        self.model = Model()
        # self.model.restore()

    def interact(self):
        # self.net.reset_state()
        while True:
            u = input(':: ')

            # check if user wants to begin new session
            if u == 'clear' or u == 'reset' or u == 'restart':
                # self.model.reset_state()
                # et = EntityTracker()
                # at = ActionTracker(et)
                print('')

            # check for exit command
            elif u == 'exit' or u == 'stop' or u == 'quit' or u == 'q':
                break

            else:
                # ENTER press : silence
                if not u:
                    u = '<SILENCE>'

                # encode
                u_ent, u_entities = self.entity_extractor.extract_entities(u)
                u_ent_features = et.context_features()

                u_emb = self.emb.encode(u)
                u_bow = self.bow_enc.encode(u)
                # concat features
                features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)
                # get action mask
                action_mask = at.action_mask()

                # forward
                prediction = self.net.forward(features, action_mask)
                print('prediction : ', prediction)
                print(u_entities)
                print('\n')
                if self.post_process(prediction, u_ent_features):
                    print('>>', 'api_call ' + u_entities['<cuisine>'] + ' ' + u_entities['<location>']
                          + ' ' + u_entities['<party_size>'] + ' ' + u_entities['<rest_type>'])

                else:
                    prediction = self.action_post_process(prediction, u_entities)
                    print('>>', self.action_templates[prediction])

                    # if all entities is satisfied and the user agree to make a reservation.
                    if all(u_ent_featur == 1 for u_ent_featur in u_ent_features) and (prediction == 10):
                        break

    def post_process(self, prediction, u_ent_features):
        if prediction == 0:
            return True
        attr_list = [9, 12, 6, 1]
        if all(u_ent_featur == 1 for u_ent_featur in u_ent_features) and prediction in attr_list:
            return True
        else:
            return False

    def action_post_process(self, prediction, u_entities):
        attr_mapping_dict = {
            9: '<cuisine>',
            12: '<location>',
            6: '<party_size>',
            1: '<rest_type>'
        }

        # find exist and non-exist entity
        exist_ent_index = [key for key, value in u_entities.items() if value != None]
        non_exist_ent_index = [key for key, value in u_entities.items() if value == None]

        # if predicted key is already in exist entity index then find non exist entity index
        # and leads the user to input non exist entity.

        if prediction in attr_mapping_dict:
            pred_key = attr_mapping_dict[prediction]
            if pred_key in exist_ent_index:
                for key, value in attr_mapping_dict.items():
                    if value == non_exist_ent_index[0]:
                        return key
            else:
                return prediction
        else:
            return prediction


if __name__ == '__main__':
    # create interactive session
    isess = InteractiveSession()
    # begin interaction
isess.interact()