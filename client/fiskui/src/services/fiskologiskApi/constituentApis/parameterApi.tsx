import { AxiosInstance } from "axios"
import { SetParameterRequest } from "../requests/setParametersRequest";

export class ParametersApi {
  private fiskologiskClient: AxiosInstance

  constructor(fiskologiskClient: AxiosInstance) {
    this.fiskologiskClient = fiskologiskClient
  }

  public async get() {
    const url =`/parameters`;
    const response = await this.fiskologiskClient.get(url);
    console.log("Received response:");
    console.log(response.data);
  }

  public async set(request: SetParameterRequest) {
    const url =`/parameters`;
    const response = await this.fiskologiskClient.post<SetParameterRequest>(url, request);
    console.log("Received response:");
    console.log(response.data);
  }
}
