export class DataService {
  constructor(fetcher=fetch){this.fetch=fetcher;this.static=false;this.catalog=[]}
  async request(url,value){
    if(this.static){
      if(url==='/api/catalog')return this.catalog;
      if(url==='/api/offline-plan'){const region=this.catalog.find(r=>r.id===value.region_id);if(!region)throw Error('Static region is unavailable');return this.json(`/packs/${region.pack_id}.json`);}
      throw Error('New coverage requires the Rust package service. This deployment contains prepared offline packages.');
    }
    try{return await this.json(url,value)}catch(error){
      if(url!=='/api/catalog')throw error;
      try{this.catalog=await this.json('/catalog.json');this.static=true;return this.catalog}catch{throw error}
    }
  }
  async json(url,value){const response=await this.fetch(url,value===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});const result=await response.json();if(!response.ok)throw Error(result.error||response.statusText);return result}
}
