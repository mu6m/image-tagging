import jwt from "jsonwebtoken";

export async function signAccessToken(payload: any) {
	return jwt.sign(payload, process.env.JWT_SECRET);
}

export async function verifyAccessToken(token: any) {
	try {
		const decodedToken = <any>jwt.verify(token, process.env.JWT_SECRET);
		return decodedToken;
	} catch (error) {
		if (error instanceof jwt.JsonWebTokenError) {
			return false;
		}
	}
}
