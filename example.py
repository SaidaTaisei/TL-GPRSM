import numpy as np
import matplotlib.pyplot as plt
import TL_GPRSM.models.GPRSM as GPRSM
import TL_GPRSM.utils.sampling as sampling
import TL_GPRSM.utils.metrics as metrics


def beam_function(length, width, height, yang_modulus, load_horizontal, load_vertical):
    displacement = (4.0*length*length*length/yang_modulus/height/width) * np.sqrt(np.square(load_vertical/height/height)+np.square(load_horizontal/width/width))
    return displacement

if __name__=="__main__":
    length = 3.0
    width = 0.2
    height = 0.1
    test_x = sampling.latin_hypercube_sampling(10000, 3, False)
    test_x = sampling.uniform_scaling(test_x, np.array([2.06e11*0.9, 5000.0*0.8, 10000.0*0.8]), np.array([2.06e11*1.1, 5000.0*1.2, 10000.0*1.2]))
    test_y = np.array([beam_function(length, width, height, test_x[i,0], test_x[i,1], test_x[i,2]) for i in range(test_x.shape[0])])[:,np.newaxis]
    
    r2s = []
    for i in range(50):
        target_x = sampling.latin_hypercube_sampling(5, 3, False)
        target_x = sampling.uniform_scaling(target_x, np.array([7.0e10*0.9, 5000.0*0.8, 10000.0*0.8]), np.array([7.0e10*1.1, 5000.0*1.2, 10000.0*1.2]))
        target_y = np.array([beam_function(length, width, height, target_x[i,0], target_x[i,1], target_x[i,2]) for i in range(target_x.shape[0])])[:,np.newaxis]
        source_x = sampling.latin_hypercube_sampling(50, 3, False)
        source_x = sampling.uniform_scaling(source_x, np.array([2.06e11*0.9, 5000.0*0.8, 10000.0*0.8]), np.array([2.06e11*1.1, 5000.0*1.2, 10000.0*1.2]))
        source_y = np.array([beam_function(length, width, height, source_x[i,0], source_x[i,1], source_x[i,2]) for i in range(source_x.shape[0])])[:,np.newaxis]
        
        gprsm = GPRSM(target_x, target_y, kernel_name="Matern52")
        gprsm.set_transfer_learning(source_x, source_y)
        gprsm.optimize(max_iter=1e4)
        contributions = gprsm.get_ard_contribution()
        print(contributions)
        test_x = sampling.latin_hypercube_sampling(10000, 3, False)
        test_x = sampling.uniform_scaling(test_x, np.array([7.0e10*0.9, 5000.0*0.8, 10000.0*0.8]), np.array([7.0e10*1.1, 5000.0*1.2, 10000.0*1.2]))
        test_y = np.array([beam_function(length, width, height, test_x[i,0], test_x[i,1], test_x[i,2]) for i in range(test_x.shape[0])])[:,np.newaxis]
        predict_y_mean, predict_y_std = gprsm.predict(test_x)
        r2 = metrics.r2_index(test_y, predict_y_mean)
        _mape = metrics.mape(test_y, predict_y_mean)
        _mae = metrics.mae(test_y, predict_y_mean)
        r2s.append(r2)
        print(r2)
        print(_mape)
        print(_mae)
    r2s = np.array(r2s)
    print("r2 mean: ", np.mean(r2s))
    print("r2 std: ", np.std(r2s))
    