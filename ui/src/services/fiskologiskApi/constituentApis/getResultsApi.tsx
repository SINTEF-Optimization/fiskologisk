import { AxiosInstance } from "axios"

export class ResultsApi {
  private fiskologiskClient: AxiosInstance

  constructor(fiskologiskClient: AxiosInstance) {
    this.fiskologiskClient = fiskologiskClient
  }

  public async get() {
    const url =`/results`;
    const response = await this.fiskologiskClient.get(url);
    console.log("Received response:");
    console.log(response.data);
    return response.data;
  }
}
