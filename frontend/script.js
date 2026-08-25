// On récupère le formulaire HTML grâce à son id="prediction-form"
const form = document.getElementById("prediction-form");

// On récupère la zone où afficher le résultat
const result = document.getElementById("predicted-price");


// On écoute l'événement "submit" du formulaire
form.addEventListener("submit", async function (event) {

    // Empêche le rechargement de la page
    event.preventDefault();


    // ----------------------------------------------------
    // 1. Construction des données envoyées à l'API
    // ----------------------------------------------------

    const data = {

        // Champs numériques
        surface: Number(
            document.getElementById("surface").value
        ),

        nb_pieces: Number(
            document.getElementById("nb_pieces").value
        ),

        etage: Number(
            document.getElementById("etage").value
        ),

        bain: Number(
            document.getElementById("bain").value
        ),


        // Champs texte / catégories
        typedebien:
            document.getElementById("typedebien").value,

        exposition:
            document.getElementById("exposition").value,


        // 1 = oui / 0 = non
        balcon: Number(
            document.getElementById("balcon").value
        ),

        logement_neuf: Number(
            document.getElementById("logement_neuf").value
        ),


        // Informations énergétiques
        dpeL:
            document.getElementById("dpeL").value,

        ges_class:
            document.getElementById("ges_class").value,


        // Informations sur l'annonce
        type_annonceur:
            document.getElementById("type_annonceur").value,

        annonce_exclusive: Number(
            document.getElementById("annonce_exclusive").value
        ),


        // Localisation
        location:
            document.getElementById("location").value
    };


    // Affiche dans la console les données envoyées
    console.log("Données envoyées à l'API :", data);


    // ----------------------------------------------------
    // 2. Affichage temporaire pendant la prédiction
    // ----------------------------------------------------

    result.textContent = "Calcul en cours...";


    try {

        // ------------------------------------------------
        // 3. Envoi de la requête POST vers FastAPI
        // ------------------------------------------------

        const response = await fetch(
            "http://127.0.0.1:8000/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(data)
            }
        );


        // ------------------------------------------------
        // 4. Lecture de la réponse JSON
        // ------------------------------------------------

        const responseData = await response.json();


        // Logs utiles pour le debug
        console.log("Status HTTP :", response.status);
        console.log("Réponse de l'API :", responseData);


        // ------------------------------------------------
        // 5. Vérification des erreurs HTTP
        // ------------------------------------------------

        if (!response.ok) {

            // Si FastAPI renvoie un message dans "detail",
            // on l'affiche
            const errorMessage =
                responseData.detail ||
                "Une erreur est survenue lors de la prédiction.";

            throw new Error(errorMessage);
        }


        // ------------------------------------------------
        // 6. Vérification de la présence du prix
        // ------------------------------------------------

        if (responseData.prix === undefined) {

            throw new Error(
                "L'API n'a pas retourné de champ 'prix'."
            );
        }


        // ------------------------------------------------
        // 7. Affichage du nouveau prix
        // ------------------------------------------------

        result.textContent =
            Number(responseData.prix).toFixed(2) + " €/m²";


    } catch (error) {

        // ------------------------------------------------
        // 8. Gestion des erreurs
        // ------------------------------------------------

        console.error(
            "Erreur lors de la prédiction :",
            error
        );

        result.textContent =
            "Erreur : " + error.message;
    }
});