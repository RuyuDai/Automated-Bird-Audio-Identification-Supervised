from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd

def run_randomforest(pcp, random_state,n_estimators):
    mid = pcp.copy()
    mid_species_list = pd.DataFrame(mid.species.unique())
    mid_species_list = mid_species_list.rename(columns={0: 'species'})
    mid_species_list = mid_species_list.to_dict()['species']
    mid_species_list = {y: x for x, y in mid_species_list.items()}
    mid['species'] = mid.species.map(mid_species_list)
    X_train, X_test, Y_train, Y_test = train_test_split(mid.drop(labels=['species'], axis=1),
                                                        mid['species'].values,
                                                        test_size=0.3,
                                                        random_state=random_state)
    model = RandomForestClassifier(n_estimators=n_estimators,
                                   random_state=random_state)
    model.fit(X_train, Y_train)
    prediction_test = model.predict(X_test)
    print('Accuracy=', metrics.accuracy_score(Y_test, prediction_test))
    feature_list = list(mid.drop(labels=['species'], axis=1).columns)
    feature_map = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
    return metrics.accuracy_score(Y_test, prediction_test), feature_map
