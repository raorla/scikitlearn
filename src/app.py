# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import os
import json
import importlib
import sys # Importé pour la redirection de stdout/stderr
import datetime # Importé pour l'exemple de sortie simple (si utilisé) et les timestamps

# --- Configuration initiale et redirection des logs ---
# Ces premiers prints vont sur le stdout/stderr original (console ou log iExec standard)
print("Initialisation du script app.py (avant redirection des logs)...")

# Récupérer le répertoire IEXEC_OUT depuis la variable d'environnement
try:
    iexec_out = os.environ['IEXEC_OUT']
    print(f"Répertoire IEXEC_OUT (depuis env var): {iexec_out}")
except KeyError:
    print("!!! ERREUR: Variable d'environnement IEXEC_OUT non trouvée !")
    print("    Utilisation de '/iexec_out' par défaut (pour tests locaux uniquement).")
    iexec_out = "/iexec_out"

# Créer le répertoire iexec_out s'il n'existe pas (utile pour tests locaux)
if not os.path.exists(iexec_out):
    print(f"Le répertoire '{iexec_out}' n'existe pas. Création pour test local.")
    try:
        os.makedirs(iexec_out, exist_ok=True)
    except Exception as e_mkdir:
        print(f"!!! ERREUR lors de la création de '{iexec_out}': {e_mkdir}")
        exit(1)
else:
    print(f"Le répertoire '{iexec_out}' existe déjà.")

# Sauvegarder les stdout/stderr originaux
original_stdout = sys.stdout
original_stderr = sys.stderr

# Définir les chemins des fichiers de log
stdout_log_path = os.path.join(iexec_out, "app_stdout.log")
stderr_log_path = os.path.join(iexec_out, "app_stderr.log")

print(f"Tentative de redirection de stdout vers: {stdout_log_path}")
print(f"Tentative de redirection de stderr vers: {stderr_log_path}")

try:
    # 'w' pour écraser le log à chaque exécution. Utiliser 'a' pour ajouter si vous préférez.
    sys.stdout = open(stdout_log_path, 'w')
    sys.stderr = open(stderr_log_path, 'w')
    # Le message suivant ira dans app_stdout.log si la redirection a réussi
    print("Redirection de stdout et stderr vers les fichiers logs réussie.", flush=True)
except Exception as e_redir:
    # Si la redirection échoue, restaurer et logguer l'erreur sur les sorties originales
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    print(f"!!! ERREUR: Impossible de rediriger stdout/stderr vers les fichiers log: {e_redir}", file=sys.stderr)
    print("    Utilisation des flux stdout/stderr standards.", file=sys.stderr)

# --- Début du script principal (les prints vont maintenant dans les fichiers log si la redirection a fonctionné) ---
print(f"--- Début du script app.py (logs redirigés) --- Timestamp: {datetime.datetime.now().isoformat()} ---")

# Afficher les variables d'environnement potentiellement pertinentes pour le débogage
print(f"Répertoire IEXEC_OUT (utilisé): {iexec_out}") # Log dans le fichier
print(f"Valeur de OPENBLAS_NUM_THREADS: {os.environ.get('OPENBLAS_NUM_THREADS', 'non définie')}")
print(f"Valeur de MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'non définie')}")
print(f"Valeur de OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'non définie')}")
print(f"Valeur de MPLCONFIGDIR: {os.environ.get('MPLCONFIGDIR', 'non définie')}")
print(f"Valeur de HOME: {os.environ.get('HOME', 'non définie')}")

# --- Importations des bibliothèques principales ---
print("Importation des bibliothèques...")
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np
    from sklearn.datasets import make_circles, make_classification, make_moons
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.inspection import DecisionBoundaryDisplay
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    print("Bibliothèques importées avec succès.")

    sklearn_version = importlib.import_module("sklearn").__version__
    print(f"Version de scikit-learn: {sklearn_version}")
    matplotlib_version = matplotlib.__version__
    print(f"Version de matplotlib: {matplotlib_version}")
    numpy_version = np.__version__
    print(f"Version de numpy: {numpy_version}")

except ImportError as e_import:
    print(f"!!! ERREUR CRITIQUE lors de l'importation des bibliothèques: {e_import}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    error_computed_json_path = os.path.join(iexec_out, 'computed.json')
    try:
        with open(error_computed_json_path, 'w+') as f:
            json.dump({"error_message": f"Échec de l'importation des bibliothèques: {str(e_import)}",
                       "details": traceback.format_exc()}, f)
        print(f"Fichier d'erreur computed.json écrit dans {error_computed_json_path}")
    except Exception as e_json_err:
        print(f"!!! ERREUR additionnelle: Impossible d'écrire le fichier computed.json d'erreur: {e_json_err}", file=sys.stderr)
    # Le script va se terminer, les logs seront fermés dans le bloc `finally` global

# --- Définition des classifieurs ---
# (Le reste du code est identique à la version précédente qui fonctionnait localement)
# ... (copiez ici toute la section "Définition des classifieurs", "Génération des jeux de données",
#      "Logique de création des graphiques", jusqu'à juste avant la fin du script)

# --- Définition des classifieurs --- (COPIÉ DE VOTRE VERSION PRÉCÉDENTE)
print("Définition des classifieurs et des jeux de données...")
names = [
    "Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
    "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
    "Naive Bayes", "QDA",
]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    AdaBoostClassifier(random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]
X_orig, y_orig = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
)
rng = np.random.RandomState(2)
X_orig += 2 * rng.uniform(size=X_orig.shape)
linearly_separable = (X_orig, y_orig)
datasets = [
    make_moons(noise=0.3, random_state=0),
    make_circles(noise=0.2, factor=0.5, random_state=1),
    linearly_separable,
]
print("Classifieurs et jeux de données définis.")

# --- Logique de création des graphiques ---
print("Démarrage de la logique de création des graphiques...")
figure = plt.figure(figsize=(27, 9))
plot_index = 1
for ds_idx, ds_data in enumerate(datasets):
    print(f"Traitement du jeu de données {ds_idx + 1}/{len(datasets)}")
    X, y = ds_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, plot_index)
    if ds_idx == 0:
        ax.set_title("Input data")
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    plot_index += 1
    print(f"  Itération sur {len(classifiers)} classifieurs pour le jeu de données {ds_idx + 1}...")
    for clf_idx, (name, clf_model) in enumerate(zip(names, classifiers)):
        print(f"    Classifieur {clf_idx + 1}/{len(classifiers)}: {name}")
        ax = plt.subplot(len(datasets), len(classifiers) + 1, plot_index)
        try:
            pipeline = make_pipeline(StandardScaler(), clf_model)
            pipeline.fit(X_train, y_train)
            score = pipeline.score(X_test, y_test)
            DecisionBoundaryDisplay.from_estimator(
                pipeline, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
            )
            ax.scatter(
                X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
            )
            ax.scatter(
                X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors="k", alpha=0.6,
            )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_idx == 0:
                ax.set_title(name)
            ax.text(
                x_max - 0.3, y_min + 0.3, ("%.2f" % score).lstrip("0"),
                size=15, horizontalalignment="right",
            )
        except Exception as e_clf:
            print(f"!!! ERREUR lors du traitement/plotting du classifieur '{name}': {e_clf}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            if ds_idx == 0:
                ax.set_title(f"{name}\n(Erreur)", color='red', fontsize=10)
            ax.text(0.5, 0.5, "Erreur pendant\nla classification",
                    ha='center', va='center', color='red', transform=ax.transAxes, fontsize=8)
        plot_index += 1
print("Logique de création des graphiques terminée.")

# --- Sauvegarde du graphique final et création de computed.json ---
output_plot_filename = "result.pdf"
output_plot_full_path = os.path.join(iexec_out, output_plot_filename)
print(f"Préparation de la sauvegarde du graphique final dans : {output_plot_full_path}")
try:
    plt.tight_layout()
    plt.savefig(fname=output_plot_full_path)
    print(f"Graphique sauvegardé avec succès dans : {output_plot_full_path}")
except Exception as e_savefig:
    print(f"!!! ERREUR CRITIQUE lors de la sauvegarde du graphique final : {e_savefig}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)

computed_json_content = {
    "deterministic-output-path": output_plot_full_path
}
computed_json_path = os.path.join(iexec_out, 'computed.json')
print(f"Préparation de l'écriture de {computed_json_path} avec le contenu : {json.dumps(computed_json_content)}")
try:
    if os.path.exists(output_plot_full_path):
        print(f"Confirmation : Le fichier de sortie déterministe '{output_plot_full_path}' existe.")
    else:
        print(f"!!! ATTENTION CRITIQUE : Le fichier de sortie déterministe déclaré '{output_plot_full_path}' N'EXISTE PAS !", file=sys.stderr)
        print("    Cela causera probablement une erreur lors de la post-computation iExec.", file=sys.stderr)
    with open(computed_json_path, 'w+') as f:
        json.dump(computed_json_content, f)
    print(f"Fichier {computed_json_path} écrit avec succès.")
except Exception as e_json_write:
    print(f"!!! ERREUR CRITIQUE lors de l'écriture de {computed_json_path} : {e_json_write}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)

# --- Logs finaux et vérification du contenu de /iexec_out ---
print(f"Vérification finale du contenu du répertoire '{iexec_out}':")
try:
    if os.path.exists(iexec_out):
        print(f"Contenu de '{iexec_out}':")
        for item in os.listdir(iexec_out):
            item_path = os.path.join(iexec_out, item)
            try:
                item_size = os.path.getsize(item_path)
                print(f"  - {item} (taille: {item_size} octets)")
            except OSError:
                print(f"  - {item} (impossible d'obtenir la taille)")
    else:
        print(f"  Le répertoire '{iexec_out}' n'existe pas à la fin du script.")
except Exception as e_listdir:
    print(f"!!! ERREUR lors de la tentative de lister le contenu de '{iexec_out}': {e_listdir}", file=sys.stderr)
# FIN DE LA SECTION COPIÉE


# --- Nettoyage final et fermeture des logs ---
# Le message suivant ira dans app_stdout.log si la redirection a fonctionné
print(f"--- Fin du script app.py (logs redirigés) --- Timestamp: {datetime.datetime.now().isoformat()} ---", flush=True)

# S'assurer que les flux de log sont vidés (flush)
if sys.stdout != original_stdout: # Si stdout a été redirigé
    try:
        sys.stdout.flush()
        sys.stdout.close()
    except Exception as e_close_stdout:
        original_stdout.write(f"Erreur en fermant le fichier log stdout: {e_close_stdout}\n")
if sys.stderr != original_stderr: # Si stderr a été redirigé
    try:
        sys.stderr.flush()
        sys.stderr.close()
    except Exception as e_close_stderr:
        original_stderr.write(f"Erreur en fermant le fichier log stderr: {e_close_stderr}\n")

# Restaurer les stdout/stderr originaux
sys.stdout = original_stdout
sys.stderr = original_stderr

# Ce dernier message ira sur le stdout original (console ou log iExec standard)
print(f"Script app.py terminé. Vérifiez '{stdout_log_path}' et '{stderr_log_path}' pour les logs détaillés (si la redirection a fonctionné).")