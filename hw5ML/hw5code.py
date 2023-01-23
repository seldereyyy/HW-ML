import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
        
def find_best_split(feature_vector: np.array, target_vector: np.array):

    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """


    thresh = np.sort(np.unique(feature_vector))

    if len(thresh) == 1:
        return None, None, None, 0

    pass
    
    lag_thresh = np.insert(thresh, 0, 0)[0:len(thresh)]

    thresholds= (thresh + lag_thresh)/2
    thresholds = thresholds[1:len(thresholds)]

    k = len(np.unique(target_vector)) # количество классов
    idx = np.argsort(feature_vector)
    target_vector_s = target_vector[idx]
    _, left_size = np.unique(feature_vector[idx], return_index=True)
    left_size = left_size[1:len(left_size)]
    right_size = len(target_vector)-left_size

    Y_ = np.divide(np.array(list(map(lambda x: np.bincount(x, minlength=k), list(map(lambda x: np.array(target_vector_s[0:x]).astype(np.int64), left_size))))).transpose(), left_size).transpose()
    Y_2= np.divide(np.array(list(map(lambda x: np.bincount(x, minlength=k), list(map(lambda x: np.array(target_vector_s[x:len(target_vector_s)]).astype(np.int64), left_size))))).transpose(), right_size).transpose()
# сори за лонгриды))) иначе memory error

    imp1 = (1-np.sum(np.power(Y_, 2), axis =1))*left_size/len(target_vector_s)
    imp2 = (1-np.sum(np.power(Y_2, 2), axis =1))*right_size/len(target_vector_s)

    ginis = -imp1 - imp2

    imp_thr = np.column_stack((thresholds, ginis))
    best = imp_thr[imp_thr[:,1] == imp_thr[:,1].max()]

    gini_best = best[0][1]
    threshold_best = best[0][0]

    return thresholds, ginis, threshold_best, gini_best



class DecisionTree(BaseEstimator):
    def __init__(self, _feature_typee, _max_depth=1000, _min_samples_split=1, _min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", _feature_typee))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_typee = _feature_typee
        self._max_depth = _max_depth
        self._min_samples_split = _min_samples_split
        self._min_samples_leaf = _min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth):

        if np.all(sub_y == sub_y[0]): # Тут надо ==
            node["type"] = "terminal"
            node["class"] = sub_y[0]



        feature_best, threshold_best, gini_best, split = None, None, None, None
        

        if ((depth <= self._max_depth) & (sub_y.shape[0] >= self._min_samples_split) ):
            for feature in range(0, sub_X.shape[1]): #по первой фиче тоже бы пройтись 
                feature_type = self._feature_typee[feature]
                categories_map = {}

                if feature_type == "real":
                    feature_vector = sub_X[:, feature]
                elif feature_type == "categorical":
                    counts = Counter(sub_X[:, feature]) 
                    clicks = Counter(sub_X[sub_y == 1, feature]) 
                    ratio = {}
                    for key, current_count in counts.items():
                        if key in clicks:
                            current_click = clicks[key]
                        else:
                            current_click = 0
                        ratio[key] = current_click / current_count # было деление на ноль
                        
                    sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1]))) # лист из отсортированных значений среднего
                    categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories))))) #тут ключ - среднее, значение - номер категории

                    feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature]))) # нужен лист и ratio
                    #print(feature_vector)
                else:
                    raise ValueError

                if len(feature_vector) <= 3: # < нужно, а не ровно 3
                    continue

                _, _, threshold, gini = find_best_split(feature_vector, sub_y)
                if threshold == None:
                    continue

                if ((gini_best is None) or (gini > gini_best)):
                    feature_best = feature
                    gini_best = gini
                    
                    split = list(feature_vector) < threshold
                    
                    if feature_type == "real":
                        threshold_best = threshold
                    elif feature_type == "categorical":
                        threshold_best = list(map(lambda x: x[0], \
                            filter(lambda x: x[1] < threshold, categories_map.items())))
                    else:
                        raise ValueError

            if (feature_best is None) | (max(sub_X[split].shape[0], sub_X[np.logical_not(split)].shape[0]) <= self._min_samples_leaf) :
                node["type"] = "terminal"
                node["class"] = Counter(sub_y).most_common(1)[0][0]

            else:
                node["type"] = "nonterminal"
                node["feature_split"] = feature_best

                if self._feature_typee[feature_best] == "real":
                    node["threshold"] = threshold_best
                elif self._feature_typee[feature_best] == "categorical":
                    node["categories_split"] = threshold_best
                else:
                    raise ValueError
                
                node["left_child"], node["right_child"] = {}, {}

                self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
                self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 2)
                           
        else:
            
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]


    def _predict_node(self, x, node) :

        if node['type'] == 'nonterminal':
            feature_type = self._feature_typee[node['feature_split']]
            if feature_type == 'real':
                if x[node['feature_split']] < node['threshold']:
                    return self._predict_node(x, node['left_child'])
                else:
                    return self._predict_node(x, node['right_child'])
            elif feature_type == 'categorical':
                if x[node['feature_split']] in node['categories_split']:
                    return self._predict_node(x, node['left_child'])
                else:
                    return self._predict_node(x, node['right_child'])
            else:
                raise ValueError
                
        else:
            return node['class']

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
