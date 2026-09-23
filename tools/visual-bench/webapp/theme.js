(()=>{
  const storageKey='navigate-color-theme';
  let choice='system';try{choice=localStorage.getItem(storageKey)||choice}catch{}
  const media=matchMedia('(prefers-color-scheme: dark)');
  const apply=()=>{document.documentElement.dataset.theme=choice==='dark'||(choice!=='light'&&media.matches)?'dark':'light'};
  apply();media.addEventListener('change',apply);
  document.addEventListener('DOMContentLoaded',()=>{
    const select=document.getElementById('theme');if(!select)return;
    select.value=['system','light','dark'].includes(choice)?choice:'system';
    select.addEventListener('change',()=>{choice=select.value;try{localStorage.setItem(storageKey,choice)}catch{}apply()});
  });
})();
