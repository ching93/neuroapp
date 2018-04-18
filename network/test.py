# from nn_model import load_model, load_mnist, evaluate
#
#
# def testfunc():
#
#     model = load_model("media/model.json")
#     model.summary()
#     (xtrain, ytrain), (xtest, ytest) = load_mnist(load_valid=False, path="media/mnist.hdf5")
#
#     print('mnist loaded')
#     model.load_weights("media/model_weights.h5")
#     print('weights loaded')
#     # print(xtest.shape)
#     # print(xtest[:100].shape)
#     (score1,score2, percent) = evaluate(model, xtest[:100], ytest[:100])
#     print("score1 - {0}, score2 - {1}, percent - {2}".format(score1,score2,percent))
#
#
# testfunc()