// On récupère le formulaire HTML grâce à son id="prediction-form"
const form = document.getElementById("prediction-form");

// On récupère la zone où on affichera le résultat
const result = document.getElementById("predicted-price");


// On écoute l'événement "submit" du formulaire
// Cette fonction sera exécutée quand on clique sur "Predict price"
form.addEventListener("submit", async function (event) {

    // Empêche le comportement normal du formulaire :
    // sans ça, la page se recharge après le clic
    event.preventDefault();


    // On construit l'objet qui sera envoyé à l'API
    const data = {

        // Pour les champs numériques :
        // .value récupère le texte du champ
        // Number(...) transforme ce texte en nombre
        surface: Number(document.getElementById("surface").value),
        nb_pieces: Number(document.getElementById("nb_pieces").value),
        etage: Number(document.getElementById("etage").value),
        bain: Number(document.getElementById("bain").value),


        // Pour les listes déroulantes / valeurs texte,
        // on récupère directement la valeur sélectionnée
        typedebien: document.getElementById("typedebien").value,
        exposition: document.getElementById("exposition").value,


        // Balcony est numérique dans ton modèle : 1 = oui, 0 = non
        balcon: Number(document.getElementById("balcon").value),

        // Ici on récupère simplement la valeur sélectionnée
        logement_neuf: document.getElementById("logement_neuf").value,


        // Informations énergétiques
        dpeL: document.getElementById("dpeL").value,
        ges_class: document.getElementById("ges_class").value,


        // Informations sur l'annonce
        type_annonceur: document.getElementById("type_annonceur").value,
        annonce_exclusive: document.getElementById("annonce_exclusive").value
    };


    // On envoie les données vers l'endpoint FastAPI /predict
    const response = await fetch("http://127.0.0.1:8000/predict", {

        // On fait une requête POST
        method: "POST",

        // On indique à FastAPI que le contenu envoyé est du JSON
        headers: {
            "Content-Type": "application/json"
        },

        // On transforme l'objet JavaScript "data"
        // en texte JSON avant de l'envoyer
        body: JSON.stringify(data)
    });


    // On récupère la réponse JSON renvoyée par FastAPI
    const responseData = await response.json();


    // Ton API renvoie par exemple :
    // { "prix": 3415.55 }
    //
    // On affiche donc la valeur "prix" dans la page
    result.textContent = responseData.prix + " €/m²";
});