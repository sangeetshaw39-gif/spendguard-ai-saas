const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';

window.CONFIG = {
    API_BASE_URL: isLocal 
        ? "http://localhost:8000" 
        : "https://spendguardai.onrender.com",
    SUPABASE_URL: "https://rquiglfgxitcyawxxlgc.supabase.co",
    SUPABASE_ANON_KEY: "sb_publishable_gGBj0551esrLIJOcZh9irA_XqD21POy"
};

