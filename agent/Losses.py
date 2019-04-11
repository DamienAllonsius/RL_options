import tensorflow as tf

class Losses():

    '''
     ' Huber loss.
     ' https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
     ' https://en.wikipedia.org/wiki/Huber_loss
    '''
    @staticmethod
    def huber_loss(y_true, y_pred, clip_delta=1.0):
      assert (y_true.shape == y_pred.shape), str(y_true.shape) + " != " + str(y_pred.shape)
      error = y_true - y_pred
      cond  = tf.keras.backend.abs(error) < clip_delta

      squared_loss = 0.5 * tf.keras.backend.square(error)
      linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

      return tf.where(cond, squared_loss, linear_loss)

    '''
     ' Same as above but returns the mean loss.
    '''
    @staticmethod
    def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
      return tf.keras.backend.mean(Losses.huber_loss(y_true, y_pred, clip_delta))


    @staticmethod
    def huber_loss_mean_weight_sampling(n_action, clip_delta=1.0):

        def loss(y_true, y_pred):
            is_weights = y_true[:, n_action]
            y_true = y_true[:, 0:n_action]

            out=is_weights * tf.keras.backend.mean( Losses.huber_loss(y_true, y_pred, clip_delta))
            #out2=is_weights * tf.losses.huber_loss(y_true, y_pred)

            return out
        return loss

    @staticmethod
    def mean_squared_error_weight_sampling(n_action, clip_delta=1.0):
        def loss(y_true, y_pred):
            is_weights = y_true[:, n_action]
            y_true = y_true[:, 0:n_action]

            out = is_weights * tf.losses.mean_squared_error(y_true, y_pred)
            # out2=is_weights * tf.losses.huber_loss(y_true, y_pred)

            return out

        return loss

    @staticmethod
    def huber_loss_importance_weight(x, y, weights):

        return tf.losses.huber_loss(labels=y, predictions=x, weights=weights, delta=1.0)