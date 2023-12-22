import { AxiosInstance } from "axios"

export class RunOptimizationApi {
  private fiskologiskClient: AxiosInstance

  constructor(fiskologiskClient: AxiosInstance) {
    this.fiskologiskClient = fiskologiskClient
  }

  public async start() {
    const url =`/start`;
    const response = await this.fiskologiskClient.post(url);
    console.log("Received response:");
    console.log(response.data);
    return response
  }
}
