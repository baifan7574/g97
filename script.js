
document.addEventListener("DOMContentLoaded", () => {
    const categories = ["dark", "office", "shower", "soft", "single", "uniform"];
    categories.forEach(category => {
        fetch(`${category}/`)
            .then(res => res.text())
            .then(html => {
                const parser = new DOMParser();
                const doc = parser.parseFromString(html, "text/html");
                const links = [...doc.querySelectorAll("a")];
                const container = document.getElementById(category);
                links.forEach(link => {
                    const href = link.getAttribute("href");
                    if (/\.(jpe?g|png|gif)$/i.test(href)) {
                        const a = document.createElement("a");
                        a.href = `${category}/${href}`;
                        a.setAttribute("data-lightbox", category);
                        const img = document.createElement("img");
                        img.src = `${category}/${href}`;
                        img.alt = href;
                        a.appendChild(img);
                        container.appendChild(a);
                    }
                });
            });
    });
});
