from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
import pandas as pd
import os
from django.views.generic import TemplateView
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from django.http import HttpResponse
from django.views import View
from django.shortcuts import render




from .forms import (
    ColumnSelectionForm,
    BivariateColumnSelectionForm,
    MultipleColumnSelectionForm,
    DataTreatmentForm,
)
from .analysis_functions import *
from .models import *
import pickle
import base64
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import AuthenticationForm
from django.contrib import messages
from datetime import datetime


def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get("username")
            password = form.cleaned_data.get("password")
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect(
                    "index"
                )  # Rediriger vers la page d'accueil ou toute autre page après la connexion
    else:
        form = AuthenticationForm()
        messages.error(request, "Nom d'utilisateur ou mot de passe incorrect.")
    return render(request, "login.html", {"form": form, "messages": messages})


#@login_required
import pandas as pd
from django.shortcuts import render
from io import StringIO

import pandas as pd
from io import StringIO
from django.shortcuts import render

import pandas as pd
from django.shortcuts import render
from io import StringIO

def data_overview_view(request):
    if "uploaded_data" not in request.session:
        # Chemin vers le fichier Excel
        static_file_path = "BD_SNI.xlsx"
        # Lire le fichier Excel dans un DataFrame
        uploaded_data = pd.read_excel(static_file_path)
        # Convertir le DataFrame en JSON et le stocker dans la session
        request.session["uploaded_data"] = uploaded_data.to_json()
    else:
        # Récupérer les données JSON depuis la session et les convertir en DataFrame
        uploaded_data_json = request.session["uploaded_data"]
        uploaded_data = pd.read_json(StringIO(uploaded_data_json))

    # Calculer la forme des données pour l'affichage
    formatted_shape = f"Nombre de lignes : {uploaded_data.shape[0]}, Nombre de colonnes : {uploaded_data.shape[1]}"

    # Formater les en-têtes en gras pour l'affichage dans <pre>
    def format_header(col):
        return f"<b>{col}</b>"  # Encapsuler les noms des colonnes dans <b>

    # Créer une copie du DataFrame avec des en-têtes en gras pour data_full
    temp_df = uploaded_data.copy()
    temp_df.columns = [format_header(col) for col in temp_df.columns]

    # Convertir le DataFrame en texte pour l'affichage dans <pre>
    data_full = temp_df.to_string()

    # Passer les données, la forme des données et data_full au template
    return render(request, "overview.html", {
        "data": uploaded_data.to_dict(orient="records"),  # Utiliser le DataFrame original
        "formatted_shape": formatted_shape,
        "data_full": data_full,  # Pour <pre id="tab">
    })


# @login_required
def plot_selection_view(request):
#     if "uploaded_data" in request.session:
#
#         uploaded_data = pd.read_json(request.session["uploaded_data"])
#
#         plots = request.session.get("plot", [])
#         target_column = request.session.get("target_column")
#         categorical_cols, numeric_cols = decompose_variables(uploaded_data)
#
#         selected_numeric_column = None
#         selected_categorical_column = None
#
#         if request.method == "POST":
#             plot_dir = "media/plots/"
#             os.makedirs(plot_dir, exist_ok=True)
#
#             # Pour les colonnes numériques
#             if "plot_numeric_distribution" in request.POST:
#                 form = ColumnSelectionForm(request.POST, columns=numeric_cols)
#                 if form.is_valid():
#                     selected_numeric_column = form.cleaned_data["column"]
#                     filename = os.path.join(plot_dir, "numeric_distribution.png")
#                     plot_numeric_distribution(uploaded_data, selected_numeric_column, filename)
#                     plot_path = f"/media/plots/numeric_distribution.png"
#                     plots.append(plot_path)
#
#             # Pour les colonnes catégoriques
#             if "plot_categorical_distribution" in request.POST:
#                 form = ColumnSelectionForm(request.POST, columns=categorical_cols)
#                 if form.is_valid():
#                     selected_categorical_column = form.cleaned_data["column"]
#                     filename = os.path.join(plot_dir, "categorical_distribution.png")
#                     plot_categorical_distribution(uploaded_data, selected_categorical_column, filename)
#                     plot_path = f"/media/plots/categorical_distribution.png"
#                     plots.append(plot_path)
#
#             if "plot_pie_chart" in request.POST:
#                 if target_column and target_column in uploaded_data.columns:
#                     # Générer le Pie Chart pour la colonne cible
#                     filename = os.path.join(plot_dir, "pie_chart.png")
#                     print("Target column for pie chart:", target_column)  # Debug
#                     plot_pie_chart(uploaded_data, target_column, filename)
#                     plot_path = f"/media/plots/pie_chart.png"
#                     plots.append(plot_path)
#
#             request.session["plots"] = plots
#
#         # Passer la colonne sélectionnée aux formulaires
#         numeric_column_selection_form = ColumnSelectionForm(columns=numeric_cols, selected_column=selected_numeric_column)
#         categorical_column_selection_form = ColumnSelectionForm(columns=categorical_cols, selected_column=selected_categorical_column)
#
#         return render(
#             request,
#             "plot.html",
#             {
#                 "numeric_column_selection_form": numeric_column_selection_form,
#                 "categorical_column_selection_form": categorical_column_selection_form,
#                 "categorical_distribution_plots": [
#                     plot for plot in plots if "categorical_distribution" in plot
#                 ],
#                 "pie_chart_plots": [plot for plot in plots if "pie_chart" in plot],
#                 "numeric_distribution_plots": [
#                     plot for plot in plots if "numeric_distribution" in plot
#                 ],
#                 "target_column": target_column,
#                 "plots": plots
#             },
#         )
#
     return redirect("overview")

# @login_required
def feature_selection_view(request):
#     if "uploaded_data" in request.session:
#         uploaded_data = pd.read_json(request.session["uploaded_data"])

#         # Récupérer la colonne cible à partir de la session
#         target_column = request.session.get("target_column")
#         message = None  # Initialisation du message à None
#         anova_results = None
#         iv_results = None
#         correlation_plot = None

#         if request.method == "POST":
#             # Vérifier si une colonne doit être supprimée
#             if "column_to_remove" in request.POST:
#                 # Suppression de colonne
#                 column_to_remove = request.POST.get("column_to_remove")
#                 if column_to_remove in uploaded_data.columns:
#                     uploaded_data = uploaded_data.drop(columns=[column_to_remove])
#                     # Mettre à jour les données dans la session après suppression
#                     request.session["uploaded_data"] = uploaded_data.to_json()
#                     message = f"La colonne {column_to_remove} a été supprimée."
#                 else:
#                     message = f"La colonne {column_to_remove} n'existe pas."

#             # Effectuer les tests ANOVA, IV, et générer la matrice de corrélation si une colonne cible est présente
#             elif target_column:
#                 anova_results = chi2_test(uploaded_data, target_column)
#                 iv_results = calculate_iv_table_with_binning(uploaded_data, target_column)
#                 correlation_matrix(uploaded_data, plot=True)
#                 correlation_plot = "/media/plots/correlation_matrix.png"
        
#         return render(
#             request,
#             "feature_selection.html",
#             {
#                 "uploaded_data": uploaded_data,
#                 "anova_results": anova_results,
#                 "iv_results": iv_results,
#                 "correlation_plot": correlation_plot,
#                 "current_column": target_column,  # Passer la colonne cible à la vue
#                 "message": message,  # Passer le message à la vue
#             },
#         )
     return redirect("upload_file")


# @login_required

def feature_selection_view(request):
    return render(request, "feature_selection.html")


# @login_required
def modeling_view(request):
#     if "uploaded_data" in request.session:
#         uploaded_data = pd.read_json(request.session["uploaded_data"])
#         target = None
#         features_selected = []
#         summary_table = None
#         reg = None
#         performance_metrics = None
#         confusion_matrix_plot = None
#         X_train = X_test = y_train = y_test = None

#         if request.method == "POST":
#             target_selection_form = ColumnSelectionForm(
#                 request.POST, columns=uploaded_data.columns
#             )
#             feature_selection_form = MultipleColumnSelectionForm(
#                 request.POST, columns=uploaded_data.columns
#             )

#             if target_selection_form.is_valid() and feature_selection_form.is_valid():
#                 target = target_selection_form.cleaned_data["column"]
#                 features_selected = feature_selection_form.cleaned_data["columns"]

#                 # Remove target from features_selected if present
#                 if target in features_selected:
#                     features_selected.remove(target)

#                 # Store target and selected features in session
#                 request.session["selected_target"] = target
#                 request.session["selected_features"] = features_selected

#                 if "model_with_balance" in request.POST:
#                     summary_table, reg, X_train, X_test, y_train, y_test = (
#                         modelwithbalance(uploaded_data, target, features_selected)
#                     )
#                     request.session["model"] = base64.b64encode(
#                         pickle.dumps(reg)
#                     ).decode("utf-8")
#                     request.session["X_train"] = X_train.to_json()
#                     request.session["X_test"] = X_test.to_json()
#                     request.session["y_train"] = y_train.to_json()
#                     request.session["y_test"] = y_test.to_json()
#                     request.session["model_columns"] = X_train.columns.tolist()

#                 elif "model_without_balance" in request.POST:
#                     summary_table, reg, X_train, X_test, y_train, y_test = (
#                         modelwithoutbalance(uploaded_data, target, features_selected)
#                     )
#                     request.session["model"] = base64.b64encode(
#                         pickle.dumps(reg)
#                     ).decode("utf-8")
#                     request.session["X_train"] = X_train.to_json()
#                     request.session["X_test"] = X_test.to_json()
#                     request.session["y_train"] = y_train.to_json()
#                     request.session["y_test"] = y_test.to_json()
#                     request.session["model_columns"] = X_train.columns.tolist()

#             # Récupération du modèle et des données d'entraînement/test stockés dans la session
#             if "model" in request.session:
#                 reg = pickle.loads(base64.b64decode(request.session["model"]))
#                 X_train = pd.read_json(request.session["X_train"])
#                 X_test = pd.read_json(request.session["X_test"])
#                 y_train = pd.read_json(request.session["y_train"], typ="series")
#                 y_test = pd.read_json(request.session["y_test"], typ="series")

#             if reg is not None:
#                 if "show_metrics" in request.POST:
#                     metrics_set = request.POST.get("metrics_set")
#                     if metrics_set == "train":
#                         performance_metrics = print_performance_metrics(
#                             reg, X_train, y_train, "Train"
#                         )
#                         confusion_matrix_plot = plot_confusion_matrix(
#                             reg, X_train, y_train, "Train"
#                         )
#                     elif metrics_set == "test":
#                         performance_metrics = print_performance_metrics(
#                             reg, X_test, y_test, "Test"
#                         )
#                         confusion_matrix_plot = plot_confusion_matrix(
#                             reg, X_test, y_test, "Test"
#                         )
#                 else:
#                     performance_metrics = None
#                     confusion_matrix_plot = None

#         else:
#             target_selection_form = ColumnSelectionForm(columns=uploaded_data.columns)
#             feature_selection_form = MultipleColumnSelectionForm(
#                 columns=uploaded_data.columns
#             )

#         return render(
#             request,
#             "modeling.html",
#             {
#                 "target_selection_form": target_selection_form,
#                 "feature_selection_form": feature_selection_form,
#                 "summary_table": (
#                     summary_table.to_dict("records")
#                     if summary_table is not None
#                     else None
#                 ),
#                 "performance_metrics": performance_metrics,
#                 "confusion_matrix_plot": confusion_matrix_plot,
#             },
#         )
    return redirect("upload_file")


@login_required
def modeling_view(request):
    # if "uploaded_data" in request.session:
#     uploaded_data = pd.read_json(request.session["uploaded_data"])
#     target_column = request.session.get("target_column")
#     summary_table = None
#     reg = None
#     performance_metrics = None
#     confusion_matrix_plot = None
#     X_train = X_test = y_train = y_test = None
#     selected_features = None  # Initialisation ici

#     if request.method == "POST" and "train_model" in request.POST:
#         if target_column and target_column in uploaded_data.columns:
#             target = target_column
#             features_selected = [col for col in uploaded_data.columns if col != target]

#             # Extract feature set and target
#             X = uploaded_data[features_selected]
#             y = uploaded_data[target]

#             # Preprocess data before feature selection
#             X_preprocessed, preprocessor = preprocess_data(X)

#             # Automatic feature selection using stepwise_selection
#             if "auto_selection" in request.POST:
#                 selected_features = stepwise_selection(
#                     pd.DataFrame(X_preprocessed, columns=preprocessor.get_feature_names_out()), y)
#                 X_preprocessed = pd.DataFrame(X_preprocessed, columns=preprocessor.get_feature_names_out())[selected_features]
                
#                 print("Forme du DataFrame transformé :", X_preprocessed.shape)
#                 # Save selected features to session for future use
#                 # Convert Index to list if needed
#                 if isinstance(selected_features, pd.Index):
#                     selected_features = selected_features.tolist()
#                 request.session["selected_features"] = selected_features
#                 print("Selected features after stepwise selection:", selected_features)

#             # Split the data
#             X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

#             # Call the appropriate modeling function
#             if "model_with_balance" in request.POST:
#                 summary_table, reg, X_train, X_test, y_train, y_test = modelwithbalance(uploaded_data, target, features_selected)
            
#             print("Shapes after balancing and encoding:")
#             print("X_train shape:", X_train.shape)
#             print("X_test shape:", X_test.shape)
#             print("y_train shape:", y_train.shape)
#             print("y_test shape:", y_test.shape)
            
#             # Check if summary_table is None and handle it
#             if summary_table is None:
#                 summary_table = pd.DataFrame()_

    return redirect("overview")

from django.shortcuts import render
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime
import os
import logging
from scipy.stats import norm
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Paramètres pour UL et RWA
rho = 0.12  # Corrélation entre actifs
z_score_999 = norm.ppf(0.999)  # Quantile pour 0.999

def score_autonomie_financiere(ratio):
    if ratio < 0:
        return 0
    elif ratio < 0.10:
        return 5
    elif ratio < 0.2:
        return 10
    elif ratio < 0.4:
        return 15
    else:
        return 20

def score_rentabilite_nette(ratio):
    if ratio < 0:
        return 0
    elif ratio < 0.05:
        return 5
    elif ratio < 0.10:
        return 10
    elif ratio < 0.20:
        return 15
    else:
        return 20

def score_liquidite_generale(ratio):
    if ratio < 0:
        return 0
    elif ratio < 1:
        return 10
    elif ratio < 2:
        return 15
    else:
        return 20

def score_endettement(ratio):
    if ratio < 0:
        return 0
    elif ratio > 5:
        return 0
    elif ratio >= 2.5:
        return 5
    else:
        return 20

def score_defaut(defaut):
    return 20 if defaut == 0 else 0

def score_anciennete(annees):
    if annees >= 35:
        return 20
    elif annees >= 25:
        return 15
    elif annees >= 10:
        return 5
    else:
        return 0

def score_reputation(rep):
    rep = str(rep).strip().lower()
    if "très bonne" in rep:
        return 20
    elif "tres bonne" in rep:
        return 20
    elif "bonne" in rep:
        return 15
    elif "moyenne" in rep:
        return 10
    elif "mauvais" in rep:
        return 5
    return 0

def score_positionnement(pos):
    pos = str(pos).strip().lower()
    if "leader" in pos:
        return 20
    elif "acteur majeur" in pos:
        return 15
    elif "acteur marginal" in pos:
        return 10
    elif "non significatif" in pos:
        return 5
    return 0
from django.utils.translation import gettext_lazy as _

def attribuer_niveau_commentaire(pd):
    try:
        pd = float(pd)
        if 0.00 <= pd <= 0.03:
            return 1, _("Excellent")
        elif 0.03 < pd <= 0.25:
            return 2, _("Very Good")
        elif 0.25 < pd <= 4.74:
            return 3, _("Good")
        elif 4.74 < pd <= 26.89:
            return 4, _("Average")
        elif 26.89 < pd <= 100.00:
            return 5, _("To Observe")
    except (ValueError, TypeError):
        logger.error(f"Invalid PD value: {pd}")
        return 1, _("Excellent")
    
def calculer_lgd(type_surete):
    if type_surete == "Garantie financière":
        return 10
    elif type_surete == "Hypothèque immobilière":
        return 30
    elif type_surete == "Nantissement sur bien":
        return 50
    elif type_surete == "Caution personnelle":
        return 60
    elif type_surete == "Sans garantie":
        return 100
    else:
        return 100  # Par défaut si le type n'est pas reconnu

def calculate_UL(row, z_999, rho):
    pd = row['pd'] / 100  # Convertir PD en décimal
    lgd = row['taux_lgd'] / 100  # Convertir LGD en décimal
    ead = row['ead']
    ul = ead * lgd * (norm.cdf((norm.ppf(pd) + np.sqrt(rho) * z_999) / np.sqrt(1 - rho)) - pd)
    return ul

def calculer_K(PD, LGD, rho):
    PD = PD / 100  # Convertir PD en décimal
    LGD = LGD / 100  # Convertir LGD en décimal
    phi_inv_PD = norm.ppf(PD)
    K = LGD * norm.cdf(1.75 * np.sqrt(rho))  # Approximation simplifiée
    return K

import locale
# Set French locale for number formatting
try:
    locale.setlocale(locale.LC_ALL, 'fr_FR.UTF-8')
except locale.Error:
    locale.setlocale(locale.LC_ALL, 'French_France.1252')

# Fonction pour formater les nombres en style français
def format_nombre(value):
    """
    Format a number (int or float) in French format with space-separated thousands
    and comma as decimal separator.
    Example: 1234567.89 -> '1 234 567,89', 1521 -> '1 521'
    """
    if isinstance(value, (int, np.integer)):
        return locale.format_string("%d", value, grouping=True)
    elif isinstance(value, (float, np.floating)):
        return locale.format_string("%.2f", value, grouping=True)
    else:
        logger.warning(f"Invalid value type for format_nombre: {type(value)}")
        return str(value)
    

def process_dataframe(request):
    try:
        # Charger les données depuis la session ou le fichier Excel
        if "uploaded_data" not in request.session:
            static_file_path = os.path.join(os.path.dirname(__file__), 'static', 'BD_SNI.xlsx')
            if not os.path.exists(static_file_path):
                logger.error(f"Excel file not found at: {static_file_path}")
                raise FileNotFoundError(f"Excel file not found at: {static_file_path}")
            df = pd.read_excel(static_file_path)
            request.session["uploaded_data"] = df.to_json()
            logger.debug("Loaded data from Excel file")
        else:
            uploaded_data_json = request.session["uploaded_data"]
            df = pd.read_json(StringIO(uploaded_data_json))
            logger.debug("Loaded data from session")

        logger.debug(f"DataFrame shape: {df.shape}")
        logger.debug(f"DataFrame columns: {df.columns.tolist()}")

        # Vérifier et ajouter les colonnes nécessaires
        required_columns = ['autonomie_financière', 'rentabilité_nette', 'liquidité_générale', 'endettement']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0
                logger.warning(f"Added missing column: {col}")

        # Vérifier les colonnes comportementales
        behavioral_columns = ['defaut', 'REPUTATION', 'POSITIONNEMENTMARCHE', 'DATE_DE_CREATION_TIERS']
        for col in behavioral_columns:
            if col not in df.columns:
                df[col] = '' if col in ['REPUTATION', 'POSITIONNEMENTMARCHE'] else 0
                logger.warning(f"Added missing column: {col}")

        # Vérifier la présence de numtiers
        if 'numtiers' not in df.columns:
            logger.error("Missing required column: numtiers")
            return pd.DataFrame()
  
        df["autonomie_financière"] = df["CAPITAUX_PROPRES"] / df["TOTAL_BILAN"]
        df["rentabilité_nette"] = df["RESULTAT_NET"] / df["CAPITAUX_PROPRES"]
        df["liquidité_générale"] = df["ACTIF_CIRCULANT"] / df["PASSIF_CIRCULANT"]
        df["endettement"] = df["DETTE_FINANCIERE"] / df["EXCEDENT_BRUT_EXPLOITATION"]

        # Appliquer les fonctions de scoring financier
        df['score_autonomie_financiere'] = df['autonomie_financière'].apply(score_autonomie_financiere)
        df['score_rentabilite_nette'] = df['rentabilité_nette'].apply(score_rentabilite_nette)
        df['score_liquidite_generale'] = df['liquidité_générale'].apply(score_liquidite_generale)
        df['score_endettement'] = df['endettement'].apply(score_endettement)

        # Calculer le score financier global
        df['score_financier'] = (
            0.25 * df['score_autonomie_financiere'] +
            0.25 * df['score_rentabilite_nette'] +
            0.25 * df['score_liquidite_generale'] +
            0.25 * df['score_endettement']
        )

        # Conversion en datetime
        df["DATE_DE_CREATION_TIERS"] = pd.to_datetime(df["DATE_DE_CREATION_TIERS"], errors='coerce')

        # Date actuelle
        date_aujourdhui = pd.Timestamp(datetime.today().date())

        # Calcul de l’ancienneté
        df["ancienneté"] = (date_aujourdhui - df["DATE_DE_CREATION_TIERS"]).dt.days / 365.25
        df["ancienneté"] = df["ancienneté"].round(1).fillna(0)

        # Appliquer les fonctions de scoring comportemental
        df["score_defaut"] = df["defaut"].apply(score_defaut)
        df["score_ancienneté"] = df["ancienneté"].apply(score_anciennete)
        df["score_reputation"] = df["REPUTATION"].apply(score_reputation)
        df["score_positionnement"] = df["POSITIONNEMENTMARCHE"].apply(score_positionnement)

        # Calculer le score comportemental total
        df["score_comportemental_total"] = (
            0.4 * df["score_defaut"] +
            0.3 * df["score_ancienneté"] +
            0.15 * df["score_reputation"] +
            0.15 * df["score_positionnement"]
        )

        # Calculer le score total global
        df["score_total_global"] = (
            df["score_financier"] * 0.6 +
            df["score_comportemental_total"] * 0.4
        )

        # Calculer PD
        df['scoreajusté'] = df['score_total_global'] - 10
        df['Score probabiliste'] = 1 / (1 + np.exp(-df['scoreajusté']))
        df['pd'] = 1 - df['Score probabiliste']
        df['pd'] = df['pd'] * 100
        if df['pd'].isna().any():
            logger.warning(f"NaN values found in pd column: {df[df['pd'].isna()].index.tolist()}")
            df['pd'] = df['pd'].fillna(0.0)

        # Calculer Classe de Risque et Commentaire
        df[['classe_de_risque', 'Commentaire_Risque']] = df['pd'].apply(
            lambda x: pd.Series(attribuer_niveau_commentaire(x), index=['classe_de_risque', 'Commentaire_Risque'])
        )

        # Calculer LGD
        if 'type de sureté' not in df.columns:
            logger.error("Missing required column: type de sureté")
            df['type de sureté'] = 'Sans garantie'
            logger.warning("Added missing column 'type de sureté' with default 'Sans garantie'")
        df['taux_lgd'] = df['type de sureté'].apply(calculer_lgd)
        if df['taux_lgd'].isna().any():
            logger.warning(f"NaN values found in taux_lgd column: {df[df['taux_lgd'].isna()].index.tolist()}")
            df['taux_lgd'] = df['taux_lgd'].fillna(100.0)

        # Calculer EAD
        if 'DETTE_FINANCIERE' not in df.columns:
            logger.error("Missing required column: DETTE_FINANCIERE")
            df['DETTE_FINANCIERE'] = 0.0
            logger.warning("Added missing column 'DETTE_FINANCIERE' with default 0.0")
        df['ccf'] = 1
        df['ead'] = df['DETTE_FINANCIERE'] * df['ccf']
        if df['ead'].isna().any():
            logger.warning(f"NaN values found in ead column: {df[df['ead'].isna()].index.tolist()}")
            df['ead'] = df['ead'].fillna(0.0)

        # Calculer ECL
        df['ECL'] = (df['pd']/100) * (df['taux_lgd']/100)* df['ead']
        if df['ECL'].isna().any():
            logger.warning(f"NaN values found in ECL column: {df[df['ECL'].isna()].index.tolist()}")
            df['ECL'] = df['ECL'].fillna(0.0)

        # Calculer UL
        df['UL'] = df.apply(lambda row: calculate_UL(row, z_score_999, rho), axis=1)
        if df['UL'].isna().any():
            logger.warning(f"NaN values found in UL column: {df[df['UL'].isna()].index.tolist()}")
            df['UL'] = df['UL'].fillna(0.0)

        # Calculer Taux de Provisionnement
        df['taux_provisionnement'] = (df['ECL'] / df['ead'] )* 100
        df['taux_provisionnement'] = df['taux_provisionnement'].replace([np.inf, -np.inf], 0.0).fillna(0.0)

        # Calculer RWA
        df['K'] = df.apply(lambda row: calculer_K(row['pd'], row['taux_lgd'], rho), axis=1)
        df['RWA'] = df['K'] * 12.5 * df['ead']
        if df['RWA'].isna().any():
            logger.warning(f"NaN values found in RWA column: {df[df['RWA'].isna()].index.tolist()}")
            df['RWA'] = df['RWA'].fillna(0.0)

        # Calculer Fonds Propre à Réserver
        df['capital_requise'] = df['RWA'] * 0.08
        if df['capital_requise'].isna().any():
            logger.warning(f"NaN values found in capital_requise column: {df[df['capital_requise'].isna()].index.tolist()}")
            df['capital_requise'] = df['capital_requise'].fillna(0.0)

        logger.debug(f"Processed DataFrame shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error processing DataFrame: {str(e)}")
        return pd.DataFrame()

from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage

import logging

logger = logging.getLogger(__name__)

def score_view(request):
    try:
        df = process_dataframe(request)
        if df.empty:
            logger.error("Empty DataFrame in score_view")
            return render(request, 'score.html', {
                'companies': [],
                'page_obj': None,
                'error': "No data available. Please place BD_SNI.xlsx in the correct directory."
            })

        companies = df[['numtiers', 'score_financier', 'score_comportemental_total', 'score_total_global']].to_dict('records')
        paginator = Paginator(companies, 20)
        page = request.GET.get('page', 1)
        try:
            page_obj = paginator.page(page)
        except PageNotAnInteger:
            page_obj = paginator.page(1)
        except EmptyPage:
            page_obj = paginator.page(paginator.num_pages)

        return render(request, 'score.html', {
            'page_obj': page_obj,
            'companies': page_obj.object_list,
        })

    except Exception as e:
        logger.error(f"Error in score_view: {str(e)}", exc_info=True)
        return render(request, 'score.html', {
            'companies': [],
            'page_obj': None,
            'error': f"An error occurred: {str(e)}"
        })

from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

def pd_view(request):
    df = process_dataframe(request)
    if df.empty:
        logger.error("Empty DataFrame in pd_view")
        return render(request, 'pd.html', {'companies': [], 'error': "No data available. Please place BD_SNI.xlsx in C:\\Users\\ikallel\\SystemeNotationInterne\\myapp\\static\\"})
    companies = df[['numtiers', 'pd', 'classe_de_risque', 'Commentaire_Risque']].to_dict('records')
    logger.debug(f"PD view companies: {len(companies)}")
    print("First few companies:", companies[:5])

        # Pagination
    paginator = Paginator(companies, 20)  # 10 companies per page
    page = request.GET.get('page')
    try:
        page_obj = paginator.page(page)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)
    
    return render(request, 'pd.html', {
        'page_obj': page_obj,
        'error': None
    })
  
from django.utils.translation import gettext_lazy as _

def translate_type_surete(type_surete):
    """Map type_surete values to translatable strings."""
    translations = {
        'Caution personnelle': _('Personal Guarantee'),
        'Garantie financière': _('Financial Guarantee'),
        'Hypothèque immobilière': _('Real Estate Mortgage'),
        'Nantissement sur bien': _('Pledge on Asset'),
        'Sans garantie': _('No Guarantee'),
    }
    return translations.get(type_surete, type_surete)

def lgd_view(request):
    df = process_dataframe(request)
    if df.empty:
        logger.error("Empty DataFrame in lgd_view")
        return render(request, 'lgd.html', {
            'companies': [],
            'page_obj': None,
            'error': "No data available. Please place BD_SNI.xlsx in C:\\Users\\ikallel\\SystemeNotationInterne\\myapp\\static\\"
        })

# Define required and optional columns
    required_columns = ['numtiers', 'taux_lgd']
    optional_columns = ['type de sureté']  # Use actual column name after debugging
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        logger.error(f"Missing required columns in DataFrame: {missing_required}")
        return render(request, 'lgd.html', {
            'companies': [],
            'page_obj': None,
            'error': f"Missing required columns: {', '.join(missing_required)}"
        })

    # Select columns, rename if present
    selected_columns = required_columns.copy()
    rename_dict = {}
    if 'type de sureté' in df.columns:
        selected_columns.append('type de sureté')
        rename_dict['type de sureté'] = 'type_surete'
    df = df[selected_columns].rename(columns=rename_dict).fillna({'type_surete': 'N/A'})
    
    # Translate type_surete
    df['type_surete'] = df['type_surete'].apply(translate_type_surete)

    companies = df.to_dict('records')
    logger.debug(f"LGD view companies: {len(companies)}")
    print("First few companies:", companies[:5])

    # Pagination
    paginator = Paginator(companies, 20)
    page = request.GET.get('page')
    try:
        page_obj = paginator.page(page)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)

    return render(request, 'lgd.html', {
        'page_obj': page_obj,
        'companies': page_obj.object_list,
        'error': None
    })

def ead_view(request):
    df = process_dataframe(request)
    if df.empty:
        logger.error("Empty DataFrame in ead_view")
        return render(request, 'ead.html', {'companies': [], 'error': "No data available. Please place BD_SNI.xlsx in C:\\Users\\ikallel\\SystemeNotationInterne\\myapp\\static\\"})
    companies = df[['numtiers', 'ccf', 'ead']].to_dict('records')
    logger.debug(f"EAD view companies: {len(companies)}")
    print("First few companies:", companies[:5])

             # Pagination
    paginator = Paginator(companies, 20)
    page = request.GET.get('page')
    try:
        page_obj = paginator.page(page)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)
    
    return render(request, 'ead.html', {
        'page_obj': page_obj,
        'error': None
    }) 



# Fonction pour convertir une figure en base64
def fig_to_base64(fig):
    """
    Convert a matplotlib figure to a base64-encoded PNG string.
    """
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close(fig)
    return base64.b64encode(image_png).decode('utf-8')


from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger


import matplotlib.pyplot as plt
import seaborn as sns
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
import logging

import matplotlib.pyplot as plt
import seaborn as sns
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
import logging
logger = logging.getLogger(__name__)

from django.utils.translation import gettext as _  # Import translation function

def vision360_view(request):
    try:
        df = process_dataframe(request)
        if df.empty:
            logger.error("Empty DataFrame in vision360_view")
            return render(request, 'vision360.html', {
                'companies': [],
                'recap': {},
                'risk_level_plot': '',
                'error': "No data available. Please place BD_SNI.xlsx in C:\\Users\\ikallel\\SystemeNotationInterne\\myapp\\static\\"
            })

        # Préparer le récapitulatif
        ead_total = df['ead'].sum()
        ecl_total = df['ECL'].sum()
        taux_prov_global = (ecl_total / ead_total)* 100 if ead_total != 0 else 0.0
        recap = {
            'nb_total_dossiers': format_nombre(len(df)),
            'ead_total': format_nombre(ead_total),
            'ecl_total': format_nombre(ecl_total),
            'ul_total': format_nombre(df['UL'].sum()),
            'rwa_total': format_nombre(df['RWA'].sum()),
            'fonds_propres_total': format_nombre(df['capital_requise'].sum()),
            'pd_moyenne': format_nombre(round(df['pd'].mean(), 2)),
            'lgd_moyenne': format_nombre(round(df['taux_lgd'].mean(), 2)),
            'taux_prov_global': format_nombre(round(taux_prov_global, 2))
        }
        logger.debug(f"Formatted recap: {recap}")

        # Générer le graphique
        fig, ax = plt.subplots(figsize=(8, 6))
        try:
            sns.countplot(
                data=df,
                x='Commentaire_Risque',
                order=[_("Excellent"), _("Very Good"), _("Good"), _("Average"), _("To observe")],
                palette='viridis',
                ax=ax
            )
            ax.set_title(_("Distribution by Risk Class"))
            ax.set_xlabel(_("Risk Class"))
            ax.set_ylabel(_("Number of Clients"))
            ax.tick_params(axis='x', rotation=30)
            risk_level_plot = fig_to_base64(fig)
        finally:
            plt.close(fig)  # Fermer la figure pour libérer la mémoire

        # Préparer les données pour l'affichage
        companies = df[[
            'numtiers', 'pd', 'classe_de_risque', 'Commentaire_Risque',
            'taux_lgd', 'ead', 'ECL', 'UL', 'taux_provisionnement', 'RWA', 'capital_requise'
        ]].to_dict('records')
        logger.debug(f"Vision 360 view companies: {len(companies)}")

        # Pagination pour le portfolio
        portfolio_paginator = Paginator(companies, 20)
        portfolio_page = request.GET.get('portfolio_page', 1)
        try:
            page_obj_portfolio = portfolio_paginator.page(portfolio_page)
        except PageNotAnInteger:
            page_obj_portfolio = portfolio_paginator.page(1)
        except EmptyPage:
            page_obj_portfolio = portfolio_paginator.page(portfolio_paginator.num_pages)

        # Pagination pour les indicateurs
        indicators_paginator = Paginator(companies, 20)
        indicators_page = request.GET.get('indicators_page', 1)
        try:
            page_obj_indicators = indicators_paginator.page(indicators_page)
        except PageNotAnInteger:
            page_obj_indicators = indicators_paginator.page(1)
        except EmptyPage:
            page_obj_indicators = indicators_paginator.page(indicators_paginator.num_pages)

        logger.info(f"Portfolio page number: {portfolio_page}, Total pages: {portfolio_paginator.num_pages}")
        logger.info(f"Indicators page number: {indicators_page}, Total pages: {indicators_paginator.num_pages}")

        return render(request, 'vision360.html', {
            'page_obj_portfolio': page_obj_portfolio,
            'page_obj_indicators': page_obj_indicators,
            'recap': recap,
            'risk_level_plot': risk_level_plot,
        })

    except Exception as e:
        logger.error(f"Error in vision360_view: {str(e)}", exc_info=True)
        return render(request, 'vision360.html', {
            'companies': [],
            'recap': {},
            'risk_level_plot': '',
            'error': f"An error occurred: {str(e)}"
        })
   

def evaluation_view(request):
     return render(request, 'evaluation.html')
    



from django.shortcuts import render
import pandas as pd
import numpy as np
from scipy.stats import norm

def fiche_client_view(request):
    df = process_dataframe(request)
    if df.empty:
        logger.error("Empty DataFrame in fiche_client_view")
        return render(request, 'fiche_client.html', {
            'message': "Aucune donnée disponible. Veuillez placer BD_SNI.xlsx dans C:\\Users\\ikallel\\SystemeNotationInterne\\myapp\\static\\"
        })

    if request.method == "POST":
        numtiers = request.POST.get("numtiers", "").strip()
        logger.debug(f"Received numtiers: '{numtiers}'")
        
        if numtiers:
            # Convertir numtiers du DataFrame en chaînes et nettoyer
            df['numtiers'] = df['numtiers'].astype(str).str.strip()
            logger.debug(f"DataFrame numtiers sample: {df['numtiers'].head().tolist()}")
            
            # Filtrer le DataFrame
            client_data = df[df["numtiers"] == numtiers]
            logger.debug(f"Filtered client_data shape: {client_data.shape}")
            
            if not client_data.empty:
                client = {
                    "numtiers": client_data["numtiers"].iloc[0],
                    "classe": client_data["classe_de_risque"].iloc[0],
                    "pd": client_data["pd"].iloc[0],
                    "lgd": client_data["taux_lgd"].iloc[0],
                    "ead": client_data["ead"].iloc[0],
                    "ECL": client_data["ECL"].iloc[0],
                    "taux_provisionnement": client_data["taux_provisionnement"].iloc[0],
                    "UL": client_data["UL"].iloc[0],
                    "RWA": client_data["RWA"].iloc[0],
                    "fonds_propres": client_data["capital_requise"].iloc[0],
                }
                logger.debug(f"Client found: {client}")
                return render(request, "fiche_client.html", {"client": client})
            else:
                logger.warning(f"No client found for numtiers: '{numtiers}'")
                return render(request, "fiche_client.html", {"message": "Client non trouvé."})
        else:
            logger.warning("No numtiers provided in POST request")
            return render(request, "fiche_client.html", {"warning": "Veuillez fournir un Numtiers du client."})
    
    logger.debug("Rendering fiche_client.html for GET request")
    return render(request, "fiche_client.html")


from django.shortcuts import render
import pandas as pd
import numpy as np
from scipy.stats import norm

def AnalyseFinanciere_view(request):
    df = process_dataframe(request)
    if df.empty:
        logger.error("Empty DataFrame in AnalyseFinanciere_view")
        return render(request, 'AnalyseFinanciere.html', {
            'message': "Aucune donnée disponible. Veuillez placer BD_SNI.xlsx dans C:\\Users\\ikallel\\SystemeNotationInterne\\myapp\\static\\"
        })

    # GET request: afficher le tableau avec les champs demandés
    selected_columns = ["numtiers", "Annee", "Raison_Sociale", "Secteurs", "Type_analyse"]
    try:
        df_display = df[selected_columns].drop_duplicates(subset=["Raison_Sociale"]).head(5)
        table_data = df_display.to_dict(orient='records')
    except KeyError as e:
        logger.error(f"Colonnes manquantes : {e}")
        table_data = [] 

    logger.debug("Rendering AnalyseFinanciere.html for GET request")
    return render(request, "AnalyseFinanciere.html", {
        "table_data": table_data
    })